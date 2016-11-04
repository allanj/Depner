package org.statnlp.allan.nner;

import static java.util.stream.Collectors.toList;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import org.statnlp.allan.depner.Config;
import org.statnlp.allan.depner.Dataset;

import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.RuntimeInterruptedException;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * This class defines a transition-based named entity recognizer which makes use
 * of a classifier powered by a neural network. The neural network accepts
 * distributed representation inputs: dense, continuous representations of
 * words, their part of speech tags, and the labels which contained in the
 * partial recognized sequence
 *
 * @author Allan Jie
 */
public class NEReconizer {

	/** A logger for this class */
	private static Redwood.RedwoodChannels log = Redwood.channels(NEReconizer.class);

	/**
	 * Words, parts of speech, and dependency relation labels which were
	 * observed in our corpus / stored in the model
	 *
	 * @see #genDictionaries(java.util.List, java.util.List)
	 */
	private List<String> knownWords, knownPos, knownLabels;

	/**
	 * Return the set of part-of-speech tags of this parser. We normalize it a
	 * bit to help it match what other parsers use.
	 *
	 * @return Set of POS tags
	 */
	public Set<String> getPosSet() {
		Set<String> foo = Generics.newHashSet(knownPos);
		// Don't really understand why these ones are there, but remove them.
		// [CDM 2016]
		foo.remove("-NULL-");
		foo.remove("-UNKNOWN-");
		foo.remove("-ROOT-");
		// but our other models do include an EOS tag
		foo.add(".$$.");
		return Collections.unmodifiableSet(foo);
	}

	/**
	 * Mapping from word / POS / dependency relation label to integer ID
	 */
	private Map<String, Integer> wordIDs, posIDs, labelIDs;

	private List<Integer> preComputed;

	/**
	 * Given a particular recongnizer configuration, this classifier will
	 * predict the best transition to make next.
	 *
	 * The {@link edu.stanford.nlp.parser.nndep.Classifier} class handles both
	 * training and inference.
	 */
	private NNERClassifier classifier;
	private NERParsingSystem system;

	private final NEConfig config;

	/**
	 * Language used to generate
	 * {@link edu.stanford.nlp.trees.GrammaticalRelation} instances.
	 */
	@SuppressWarnings("unused")
	private final Language language;

	NEReconizer() {
		this(new Properties());
	}

	public NEReconizer(Properties properties) {
		config = new NEConfig(properties);

		// Convert Languages.Language instance to
		// GrammaticalLanguage.Language
		this.language = config.language;
	}

	/**
	 * Get an integer ID for the given word. This ID can be used to index into
	 * the embeddings {@link NNERClassifier#E}.
	 *
	 * @return An ID for the given word, or an ID referring to a generic
	 *         "unknown" word if the word is unknown
	 */
	public int getWordID(String s) {
		return wordIDs.containsKey(s) ? wordIDs.get(s) : wordIDs.get(NEConfig.UNKNOWN);
	}

	public int getPosID(String s) {
		return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(NEConfig.UNKNOWN);
	}

	public int getLabelID(String s) {
		return labelIDs.containsKey(s) ? labelIDs.get(s) : labelIDs.get(Config.UNKNOWN);
	}

	public List<Integer> getFeatures(NEConfiguration c) {
		// Presize the arrays for very slight speed gain. Hardcoded, but so is
		// the current feature list.
		// Not specify the initial capacity will make the list dynamically
		// resize
		List<Integer> fWord = new ArrayList<>(6);
		List<Integer> fPos = new ArrayList<>(6);
		List<Integer> fLabel = new ArrayList<>(3);
		for (int j = 2; j >= 0; --j) {
			int index = c.getStack(j);
			fWord.add(getWordID(c.getWord(index)));
			fPos.add(getPosID(c.getPOS(index)));
			fLabel.add(getLabelID(c.getLabel(index)));
		}
		for (int j = 0; j <= 2; ++j) {
			int index = c.getBuffer(j);
			fWord.add(getWordID(c.getWord(index)));
			fPos.add(getPosID(c.getPOS(index)));
		}

		List<Integer> feature = new ArrayList<>(15);
		feature.addAll(fWord);
		feature.addAll(fPos);
		feature.addAll(fLabel);
		return feature;
	}

	private static final int POS_OFFSET = 6;
	private static final int NE_OFFSET = 12;

	private int[] getFeatureArray(NEConfiguration c) {
		int[] feature = new int[NEConfig.numTokens]; // positions 0-5 hold fWord, 6-11 hold fPos, 12-14 hold fLabel 6+6+3
		for (int j = 2; j >= 0; --j) {
			int index = c.getStack(j);
			feature[2 - j] = getWordID(c.getWord(index));
			feature[POS_OFFSET + (2 - j)] = getPosID(c.getPOS(index));
		}
		for (int j = 0; j <= 2; ++j) {
			int index = c.getBuffer(j);
			feature[3 + j] = getWordID(c.getWord(index));
			feature[POS_OFFSET + 3 + j] = getPosID(c.getPOS(index));
		}
		for (int j = 2; j >= 0; --j) {
			int index = c.getStack(j);
			feature[NE_OFFSET + (2 - j)] = getLabelID(c.getLabel(index));
		}
		return feature;
	}

	public Dataset genTrainExamples(List<Sequence> sents, List<Sequence> ners) {
		int numTrans = system.numTransitions();
		Dataset ret = new Dataset(NEConfig.numTokens, numTrans);

		Counter<Integer> tokPosCount = new IntCounter<>();
		log.info(NEConfig.SEPARATOR);
		log.info("Generate training examples...");

		for (int i = 0; i < sents.size(); ++i) {
			if (i > 0) {
				if (i % 1000 == 0)
					log.info(i + " ");
				if (i % 10000 == 0 || i == sents.size() - 1)
					log.info();
			}
			NEConfiguration c = system.initialConfiguration(sents.get(i));
			while (!system.isTerminal(c)) {
				String oracle = system.getOracle(c, ners.get(i));
				List<Integer> feature = getFeatures(c);
				List<Integer> label = new ArrayList<>();
				for (int j = 0; j < numTrans; ++j) {
					String str = system.transitions.get(j);
					if (str.equals(oracle))
						label.add(1); // is oracle
					else if (system.canApply(c, str))
						label.add(0); //can apply but not oracle action
					else
						label.add(-1);
				}

				ret.addExample(feature, label);
				for (int j = 0; j < feature.size(); ++j)
					tokPosCount.incrementCount(feature.get(j) * feature.size() + j); //for the precomputation trick
				system.apply(c, oracle);
			}
		}
		log.info("#Train Examples: " + ret.n);

		List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);
		preComputed = new ArrayList<>(sortedTokens.subList(0, Math.min(config.numPreComputed, sortedTokens.size())));

		return ret;
	}

	/**
	 * Generate unique integer IDs for all known words / part-of-speech tags /
	 * dependency relation labels.
	 *
	 * All three of the aforementioned types are assigned IDs from a continuous
	 * range of integers; all IDs 0 <= ID < n_w are word IDs, all IDs n_w <= ID
	 * < n_w + n_pos are POS tag IDs, and so on.
	 */
	private void generateIDs() {
		wordIDs = new HashMap<>();
		posIDs = new HashMap<>();
		labelIDs = new HashMap<>();

		int index = 0;
		for (String word : knownWords)
			wordIDs.put(word, (index++));
		for (String pos : knownPos)
			posIDs.put(pos, (index++));
		for (String label : knownLabels)
			labelIDs.put(label, (index++));
	}

	/**
	 * Scan a corpus and store all words, part-of-speech tags, and dependency
	 * relation labels observed. Prepare other structures which support word /
	 * POS / label lookup at train- / run-time.
	 */
	private void genDictionaries(List<Sequence> sents, List<Sequence> ners) {
		// Collect all words (!), etc. in lists, tacking on one sentence
		// after the other
		List<String> word = new ArrayList<>();
		List<String> pos = new ArrayList<>();
		List<String> label = new ArrayList<>();

		for (Sequence sentence : sents) {

			for(int i = 0; i < sentence.size(); i++){
				word.add(sentence.get(i)[0]);
				pos.add(sentence.get(i)[1]);
			}
		}

		for (Sequence ner : ners) {
			for(int i = 0; i < ner.size(); i++){
				label.add(ner.get(i)[0]);
			}
		}
		
		// Generate "dictionaries," possibly with frequency cutoff
		knownWords = NEUtil.generateDict(word, config.wordCutOff);
		knownPos = NEUtil.generateDict(pos);
		knownLabels = NEUtil.generateDict(label);


		knownWords.add(0, NEConfig.UNKNOWN);
		knownWords.add(1, NEConfig.NULL);
		knownWords.add(2, NEConfig.ROOT);

		knownPos.add(0, NEConfig.UNKNOWN);
		knownPos.add(1, NEConfig.NULL);
		knownPos.add(2, NEConfig.ROOT);

		knownLabels.add(0, NEConfig.NULL);
		generateIDs();

		log.info(NEConfig.SEPARATOR);
		log.info("#Word: " + knownWords.size());
		log.info("#POS:" + knownPos.size());
		log.info("#Label: " + knownLabels.size());
	}

	public void writeModelFile(String modelFile) {
		try {
			double[][] W1 = classifier.getW1();
			double[] b1 = classifier.getb1();
			double[][] W2 = classifier.getW2();
			double[][] E = classifier.getE();

			Writer output = IOUtils.getPrintWriter(modelFile);

			output.write("dict=" + knownWords.size() + "\n");
			output.write("pos=" + knownPos.size() + "\n");
			output.write("label=" + knownLabels.size() + "\n");
			output.write("embeddingSize=" + E[0].length + "\n");
			output.write("hiddenSize=" + b1.length + "\n");
			output.write("numTokens=" + (W1[0].length / E[0].length) + "\n");
			output.write("preComputed=" + preComputed.size() + "\n");

			int index = 0;

			// First write word / POS / label embeddings
			for (String word : knownWords) {
				index = writeEmbedding(E[index], output, index, word);
			}
			for (String pos : knownPos) {
				index = writeEmbedding(E[index], output, index, pos);
			}
			for (String label : knownLabels) {
				index = writeEmbedding(E[index], output, index, label);
			}

			// Now write classifier weights
			for (int j = 0; j < W1[0].length; ++j)
				for (int i = 0; i < W1.length; ++i) {
					output.write(String.valueOf(W1[i][j]));
					if (i == W1.length - 1)
						output.write("\n");
					else
						output.write(" ");
				}
			for (int i = 0; i < b1.length; ++i) {
				output.write(String.valueOf(b1[i]));
				if (i == b1.length - 1)
					output.write("\n");
				else
					output.write(" ");
			}
			for (int j = 0; j < W2[0].length; ++j)
				for (int i = 0; i < W2.length; ++i) {
					output.write(String.valueOf(W2[i][j]));
					if (i == W2.length - 1)
						output.write("\n");
					else
						output.write(" ");
				}

			// Finish with pre-computation info
			for (int i = 0; i < preComputed.size(); ++i) {
				output.write(String.valueOf(preComputed.get(i)));
				if ((i + 1) % 100 == 0 || i == preComputed.size() - 1)
					output.write("\n");
				else
					output.write(" ");
			}

			output.close();
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}
	}

	private static int writeEmbedding(double[] doubles, Writer output, int index, String word) throws IOException {
		output.write(word);
		for (double aDouble : doubles) {
			output.write(" " + aDouble);
		}
		output.write("\n");
		index = index + 1;
		return index;
	}

	/**
	 * Convenience method; see
	 * {@link #loadFromModelFile(String, java.util.Properties)}.
	 *
	 * @see #loadFromModelFile(String, java.util.Properties)
	 */
	public static NEReconizer loadFromModelFile(String modelFile) {
		return loadFromModelFile(modelFile, null);
	}

	/**
	 * Load a saved parser model.
	 *
	 * @param modelFile
	 *            Path to serialized model (may be GZipped)
	 * @param extraProperties
	 *            Extra test-time properties not already associated with model
	 *            (may be null)
	 *
	 * @return Loaded and initialized (see {@link #initialize(boolean)} model
	 */
	public static NEReconizer loadFromModelFile(String modelFile, Properties extraProperties) {
		NEReconizer parser = extraProperties == null ? new NEReconizer() : new NEReconizer(extraProperties);
		parser.loadModelFile(modelFile, false);
		return parser;
	}

	/**
	 * Load a parser model file, printing out some messages about the grammar in
	 * the file.
	 *
	 * @param modelFile
	 *            The file (classpath resource, etc.) to load the model from.
	 */
	public void loadModelFile(String modelFile) {
		loadModelFile(modelFile, true);
	}

	private void loadModelFile(String modelFile, boolean verbose) {
		Timing t = new Timing();
		try {

			log.info("Loading depparse model file: " + modelFile + " ... ");
			String s;
			BufferedReader input = IOUtils.readerFromString(modelFile);

			s = input.readLine();
			int nDict = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int nPOS = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int nLabel = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int eSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int hSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int nTokens = Integer.parseInt(s.substring(s.indexOf('=') + 1));
			s = input.readLine();
			int nPreComputed = Integer.parseInt(s.substring(s.indexOf('=') + 1));

			knownWords = new ArrayList<>();
			knownPos = new ArrayList<>();
			knownLabels = new ArrayList<>();
			double[][] E = new double[nDict + nPOS + nLabel][eSize];
			String[] splits;
			int index = 0;

			for (int k = 0; k < nDict; ++k) {
				s = input.readLine();
				splits = s.split(" ");
				knownWords.add(splits[0]);
				for (int i = 0; i < eSize; ++i)
					E[index][i] = Double.parseDouble(splits[i + 1]);
				index = index + 1;
			}
			for (int k = 0; k < nPOS; ++k) {
				s = input.readLine();
				splits = s.split(" ");
				knownPos.add(splits[0]);
				for (int i = 0; i < eSize; ++i)
					E[index][i] = Double.parseDouble(splits[i + 1]);
				index = index + 1;
			}
			for (int k = 0; k < nLabel; ++k) {
				s = input.readLine();
				splits = s.split(" ");
				knownLabels.add(splits[0]);
				for (int i = 0; i < eSize; ++i)
					E[index][i] = Double.parseDouble(splits[i + 1]);
				index = index + 1;
			}
			generateIDs();

			double[][] W1 = new double[hSize][eSize * nTokens];
			for (int j = 0; j < W1[0].length; ++j) {
				s = input.readLine();
				splits = s.split(" ");
				for (int i = 0; i < W1.length; ++i)
					W1[i][j] = Double.parseDouble(splits[i]);
			}

			double[] b1 = new double[hSize];
			s = input.readLine();
			splits = s.split(" ");
			for (int i = 0; i < b1.length; ++i)
				b1[i] = Double.parseDouble(splits[i]);

			double[][] W2 = new double[nLabel * 2 - 1][hSize];
			for (int j = 0; j < W2[0].length; ++j) {
				s = input.readLine();
				splits = s.split(" ");
				for (int i = 0; i < W2.length; ++i)
					W2[i][j] = Double.parseDouble(splits[i]);
			}

			preComputed = new ArrayList<>();
			while (preComputed.size() < nPreComputed) {
				s = input.readLine();
				splits = s.split(" ");
				for (String split : splits) {
					preComputed.add(Integer.parseInt(split));
				}
			}
			input.close();
			config.hiddenSize = hSize;
			config.embeddingSize = eSize;
			classifier = new NNERClassifier(config, E, W1, b1, W2, preComputed);
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}

		// initialize the loaded parser
		initialize(verbose);
		t.done("Initializing dependency parser");
	}

	// TODO this should be a function which returns the embeddings array +
	// embedID
	// otherwise the class needlessly carries around the extra baggage of
	// `embeddings`
	// (never again used) for the entire training process
	private double[][] readEmbedFile(String embedFile, Map<String, Integer> embedID) {

		double[][] embeddings = null;
		if (embedFile != null) {
			BufferedReader input = null;
			try {
				input = IOUtils.readerFromString(embedFile);
				List<String> lines = new ArrayList<>();
				for (String s; (s = input.readLine()) != null;) {
					lines.add(s);
				}

				int nWords = lines.size();
				String[] splits = lines.get(0).split("\\s+");

				int dim = splits.length - 1;
				embeddings = new double[nWords][dim];
				log.info("Embedding File " + embedFile + ": #Words = " + nWords + ", dim = " + dim);

				if (dim != config.embeddingSize)
					throw new IllegalArgumentException(
							"The dimension of embedding file does not match config.embeddingSize");

				for (int i = 0; i < lines.size(); ++i) {
					splits = lines.get(i).split("\\s+");
					embedID.put(splits[0], i);
					for (int j = 0; j < dim; ++j)
						embeddings[i][j] = Double.parseDouble(splits[j + 1]);
				}
			} catch (IOException e) {
				throw new RuntimeIOException(e);
			} finally {
				IOUtils.closeIgnoringExceptions(input);
			}
			embeddings = NEUtil.scaling(embeddings, 0, 1.0);
		}
		return embeddings;
	}

	/**
	 * Train a new dependency parser model.
	 *
	 * @param trainFile
	 *            Training data
	 * @param devFile
	 *            Development data (used for regular UAS evaluation of model)
	 * @param modelFile
	 *            String to which model should be saved
	 * @param embedFile
	 *            File containing word embeddings for words used in training
	 *            corpus
	 * TODO: modify the accuracy to get the f-score 
	 */
	public void train(String trainFile, String devFile, String modelFile, String embedFile, String preModel) {
		log.info("Train File: " + trainFile);
		log.info("Dev File: " + devFile);
		log.info("Model File: " + modelFile);
		log.info("Embedding File: " + embedFile);
		log.info("Pre-trained Model File: " + preModel);

		List<Sequence> trainSents = new ArrayList<>();
		List<Sequence> trainNEs = new ArrayList<>();
		NEUtil.loadConllFile(trainFile, trainSents, trainNEs, config.unlabeled, config.cPOS);
		NEUtil.printNERStats("Train", trainNEs);

		List<Sequence> devSents = new ArrayList<>();
		List<Sequence> devNERs = new ArrayList<>();
		if (devFile != null) {
			NEUtil.loadConllFile(devFile, devSents, devNERs, config.unlabeled, config.cPOS);
			NEUtil.printNERStats("Dev", devNERs);
		}
		genDictionaries(trainSents, trainNEs);

		// NOTE: remove -NULL-, and the pass it to ParsingSystem
		List<String> lDict = new ArrayList<>(knownLabels);
		lDict.remove(0);
		system = new NEStandard(config.tlp, lDict, true);

		// Initialize a classifier; prepare for training
		setupClassifierForTraining(trainSents, trainNEs, embedFile, preModel);

		log.info(NEConfig.SEPARATOR);
		config.printParameters();

		long startTime = System.currentTimeMillis();
		/**
		 * Track the best accracy performance we've seen.
		 */
		double bestAcc = 0;

		for (int iter = 0; iter < config.maxIter; ++iter) {
			log.info("##### Iteration " + iter);

			NNERClassifier.Cost cost = classifier.computeCostFunction(config.batchSize, config.regParameter,
					config.dropProb);
			log.info("Cost = " + cost.getCost() + ", Correct(%) = " + cost.getPercentCorrect());
			classifier.takeAdaGradientStep(cost, config.adaAlpha, config.adaEps);

			log.info("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");

			// UAS evaluation
			if (devFile != null && iter % config.evalPerIter == 0) {
				// Redo precomputation with updated weights. This is only
				// necessary because we're updating weights -- for normal
				// prediction, we just do this once in #initialize
				classifier.preCompute();

				List<Sequence> predicted = devSents.stream().map(this::predictInner).collect(toList());

				//so far we can maximize the accuracy here.
				double acc = system.getAcc(devSents, predicted, devNERs);
//				there is no UAS
//				double uas = config.noPunc ? system.getUASnoPunc(devSents, predicted, devNERs)
//						: system.getUAS(devSents, predicted, devNERs);
//				log.info("UAS: " + uas);

				if (config.saveIntermediate && acc > bestAcc) {
					System.err.printf("Exceeds best previous UAS of %f. Saving model file..%n", bestAcc);
					bestAcc = acc;
					writeModelFile(modelFile);
				}
			}

			// Clear gradients
			if (config.clearGradientsPerIter > 0 && iter % config.clearGradientsPerIter == 0) {
				log.info("Clearing gradient histories..");
				classifier.clearGradientHistories();
			}
		}

		classifier.finalizeTraining();

		if (devFile != null) {
			// Do final UAS evaluation and save if final model beats the
			// best intermediate one
			List<Sequence> predicted = devSents.stream().map(this::predictInner).collect(toList());
			double acc = system.getAcc(devSents, predicted, devNERs);

			if (acc > bestAcc) {
				System.err.printf("Final model UAS: %f%n", acc);
				System.err.printf("Exceeds best previous UAS of %f. Saving model file..%n", bestAcc);

				writeModelFile(modelFile);
			}
		} else {
			writeModelFile(modelFile);
		}
	}

	/**
	 * @see #train(String, String, String, String, String)
	 */
	public void train(String trainFile, String devFile, String modelFile, String embedFile) {
		train(trainFile, devFile, modelFile, embedFile, null);
	}

	/**
	 * @see #train(String, String, String, String)
	 */
	public void train(String trainFile, String devFile, String modelFile) {
		train(trainFile, devFile, modelFile, null);
	}

	/**
	 * @see #train(String, String, String)
	 */
	public void train(String trainFile, String modelFile) {
		train(trainFile, null, modelFile);
	}

	/**
	 * Prepare a classifier for training with the given dataset.
	 */
	private void setupClassifierForTraining(List<Sequence> trainSents, List<Sequence> trainNERs, String embedFile,
			String preModel) {
		double[][] E = new double[knownWords.size() + knownPos.size() + knownLabels.size()][config.embeddingSize];
		double[][] W1 = new double[config.hiddenSize][config.embeddingSize * NEConfig.numTokens];
		double[] b1 = new double[config.hiddenSize];
		double[][] W2 = new double[system.numTransitions()][config.hiddenSize];

		// Randomly initialize weight matrices / vectors
		Random random = NEUtil.getRandom();
		for (int i = 0; i < W1.length; ++i)
			for (int j = 0; j < W1[i].length; ++j)
				W1[i][j] = random.nextDouble() * 2 * config.initRange - config.initRange;

		for (int i = 0; i < b1.length; ++i)
			b1[i] = random.nextDouble() * 2 * config.initRange - config.initRange;

		for (int i = 0; i < W2.length; ++i)
			for (int j = 0; j < W2[i].length; ++j)
				W2[i][j] = random.nextDouble() * 2 * config.initRange - config.initRange;

		// Read embeddings into `embedID`, `embeddings`
		Map<String, Integer> embedID = new HashMap<>();
		double[][] embeddings = readEmbedFile(embedFile, embedID);

		// Try to match loaded embeddings with words in dictionary
		int foundEmbed = 0;
		for (int i = 0; i < E.length; ++i) {
			int index = -1;
			if (i < knownWords.size()) {
				String str = knownWords.get(i);
				// NOTE: exact match first, and then try lower case..
				if (embedID.containsKey(str))
					index = embedID.get(str);
				else if (embedID.containsKey(str.toLowerCase()))
					index = embedID.get(str.toLowerCase());
			}
			if (index >= 0) {
				++foundEmbed;
				System.arraycopy(embeddings[index], 0, E[i], 0, E[i].length);
			} else {
				for (int j = 0; j < E[i].length; ++j)
					// E[i][j] = random.nextDouble() * config.initRange * 2 -
					// config.initRange;
					// E[i][j] = random.nextDouble() * 0.2 - 0.1;
					// E[i][j] = random.nextGaussian() * Math.sqrt(0.1);
					E[i][j] = random.nextDouble() * 0.02 - 0.01;
			}
		}
		log.info("Found embeddings: " + foundEmbed + " / " + knownWords.size());

		if (preModel != null) {
			try {
				log.info("Loading pre-trained model file: " + preModel + " ... ");
				String s;
				BufferedReader input = IOUtils.readerFromString(preModel);

				s = input.readLine();
				int nDict = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();
				int nPOS = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();
				int nLabel = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();
				int eSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();
				int hSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();
				int nTokens = Integer.parseInt(s.substring(s.indexOf('=') + 1));
				s = input.readLine();

				String[] splits;
				for (int k = 0; k < nDict; ++k) {
					s = input.readLine();
					splits = s.split(" ");
					if (wordIDs.containsKey(splits[0]) && eSize == config.embeddingSize) {
						int index = getWordID(splits[0]);
						for (int i = 0; i < eSize; ++i)
							E[index][i] = Double.parseDouble(splits[i + 1]);
					}
				}

				for (int k = 0; k < nPOS; ++k) {
					s = input.readLine();
					splits = s.split(" ");
					if (posIDs.containsKey(splits[0]) && eSize == config.embeddingSize) {
						int index = getPosID(splits[0]);
						for (int i = 0; i < eSize; ++i)
							E[index][i] = Double.parseDouble(splits[i + 1]);
					}
				}

				for (int k = 0; k < nLabel; ++k) {
					s = input.readLine();
					splits = s.split(" ");
					if (labelIDs.containsKey(splits[0]) && eSize == config.embeddingSize) {
						int index = getLabelID(splits[0]);
						for (int i = 0; i < eSize; ++i)
							E[index][i] = Double.parseDouble(splits[i + 1]);
					}
				}

				boolean copyLayer1 = hSize == config.hiddenSize && config.embeddingSize == eSize
						&& NEConfig.numTokens == nTokens;
				if (copyLayer1) {
					log.info("Copying parameters W1 && b1...");
				}
				for (int j = 0; j < eSize * nTokens; ++j) {
					s = input.readLine();
					if (copyLayer1) {
						splits = s.split(" ");
						for (int i = 0; i < hSize; ++i)
							W1[i][j] = Double.parseDouble(splits[i]);
					}
				}

				s = input.readLine();
				if (copyLayer1) {
					splits = s.split(" ");
					for (int i = 0; i < hSize; ++i)
						b1[i] = Double.parseDouble(splits[i]);
				}

				boolean copyLayer2 = (nLabel * 2 - 1 == system.numTransitions()) && hSize == config.hiddenSize;
				if (copyLayer2)
					log.info("Copying parameters W2...");
				for (int j = 0; j < hSize; ++j) {
					s = input.readLine();
					if (copyLayer2) {
						splits = s.split(" ");
						for (int i = 0; i < nLabel * 2 - 1; ++i)
							W2[i][j] = Double.parseDouble(splits[i]);
					}
				}
				input.close();
			} catch (IOException e) {
				throw new RuntimeIOException(e);
			}
		}
		Dataset trainSet = genTrainExamples(trainSents, trainNERs);
		classifier = new NNERClassifier(config, trainSet, E, W1, b1, W2, preComputed);
	}

	/**
	 * Determine the named entity sequence given the input sequence
	 * <p>
	 * This "inner" method returns a structure unique to this package; 
	 */
	private Sequence predictInner(Sequence sentence) {
		int numTrans = system.numTransitions();

		NEConfiguration c = system.initialConfiguration(sentence);
		while (!system.isTerminal(c)) {
			if (Thread.interrupted()) { // Allow interrupting
				throw new RuntimeInterruptedException();
			}
			double[] scores = classifier.computeScores(getFeatureArray(c));

			double optScore = Double.NEGATIVE_INFINITY;
			String optTrans = null;

			for (int j = 0; j < numTrans; ++j) {
				if (scores[j] > optScore && system.canApply(c, system.transitions.get(j))) {
					optScore = scores[j];
					optTrans = system.transitions.get(j);
				}
			}
			system.apply(c, optTrans);
		}
		return c.ners;
	}

	/**
	 * Determine the dependency parse of the given sentence using the loaded
	 * model. You must first load a parser before calling this method.
	 *
	 * @throws java.lang.IllegalStateException
	 *             If parser has not yet been loaded and initialized (see
	 *             {@link #initialize(boolean)}
	 */
	public Sequence predict(Sequence sentence) {
		if (system == null)
			throw new IllegalStateException("Parser has not been  " + "loaded and initialized; first load a model.");
		Sequence result = predictInner(sentence);
		return result;
	}



	/**
	 * Run the parser in the modelFile on a testFile and perhaps save output.
	 *
	 * @param testFile
	 *            File to parse. In CoNLL-X format. Assumed to have gold answers
	 *            included.
	 * @param outFile
	 *            File to write results to in CoNLL-X format. If null, no output
	 *            is written
	 * @return The LAS score on the dataset
	 */
	public double testCoNLL(String testFile, String outFile, String evalFile) {
		log.info("Test File: " + testFile);
		Timing timer = new Timing();
		List<Sequence> testSents = new ArrayList<>();
		List<Sequence> testNERs = new ArrayList<>();
		NEUtil.loadConllFile(testFile, testSents, testNERs, config.unlabeled, config.cPOS);

		// count how much to parse
		int numWords = 0;
		int numOOVWords = 0;
		int numSentences = 0;
		for (Sequence testSent : testSents) {
			numSentences += 1;
			for (int i = 0; i< testSent.size(); i++) {
				String word = testSent.get(i)[0];
				numWords += 1;
				if (!wordIDs.containsKey(word))
					numOOVWords += 1;
			}
		}
		System.err.printf("OOV Words: %d / %d = %.2f%%\n", numOOVWords, numWords, numOOVWords * 100.0 / numWords);

		List<Sequence> predicted = testSents.stream().map(this::predictInner).collect(toList());
		Map<String, Double> result = system.evaluate(testSents, predicted, testNERs, evalFile);

		double fscore = result.get("fscore");
		System.err.printf("F-score = %.6f%n", fscore);

		long millis = timer.stop();
		double wordspersec = numWords / (((double) millis) / 1000);
		double sentspersec = numSentences / (((double) millis) / 1000);
		System.err.printf("%s parsed %d words in %d sentences in %.1fs at %.1f w/s, %.1f sent/s.%n",
				StringUtils.getShortClassName(this), numWords, numSentences, millis / 1000.0, wordspersec, sentspersec);

		if (outFile != null) {
			NEUtil.writeConllFile(outFile, testSents, predicted);
		}
		return fscore;
	}

	/**
	 * Prepare for parsing after a model has been loaded.
	 */
	private void initialize(boolean verbose) {
		if (knownLabels == null)
			throw new IllegalStateException("Model has not been loaded or trained");

		// NOTE: remove -NULL-, and then pass the label set to the ParsingSystem
		List<String> lDict = new ArrayList<>(knownLabels);
		lDict.remove(0);

		system = new NEStandard(config.tlp, lDict, verbose);

		// Pre-compute matrix multiplications
		if (config.numPreComputed > 0) {
			classifier.preCompute();
		}
	}

	/**
	 * Explicitly specifies the number of arguments expected with particular
	 * command line options.
	 */
	private static final Map<String, Integer> numArgs = new HashMap<>();
	static {
		numArgs.put("textFile", 1);
		numArgs.put("outFile", 1);
	}

	/**
	 * A main program for training, testing and using the parser.
	 *
	 * <p>
	 * You can use this program to train new parsers from treebank data,
	 * evaluate on test treebank data, or parse raw text input.
	 *
	 * <p>
	 * Sample usages:
	 * <ul>
	 * <li><strong>Train a parser with CoNLL treebank data:</strong>
	 * {@code java edu.stanford.nlp.parser.nndep.DependencyParser -trainFile trainPath -devFile devPath -embedFile wordEmbeddingFile -embeddingSize wordEmbeddingDimensionality -model modelOutputFile.txt.gz}
	 * </li>
	 * <li><strong>Parse raw text from a file:</strong>
	 * {@code java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile rawTextToParse -outFile dependenciesOutputFile.txt}
	 * </li>
	 * <li><strong>Parse raw text from standard input, writing to standard
	 * output:</strong>
	 * {@code java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile - -outFile -}
	 * </li>
	 * </ul>
	 *
	 * <p>
	 * See below for more information on all of these training / test options
	 * and more.
	 *
	 * <p>
	 * Input / output options:
	 * <table>
	 * <tr>
	 * <th>Option</th>
	 * <th>Required for training</th>
	 * <th>Required for testing / parsing</th>
	 * <th>Description</th>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;devFile</tt></td>
	 * <td>Optional</td>
	 * <td>No</td>
	 * <td>Path to a development-set treebank in
	 * <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a>. If
	 * provided, the</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;embedFile</tt></td>
	 * <td>Optional (highly recommended!)</td>
	 * <td>No</td>
	 * <td>A word embedding file, containing distributed representations of
	 * English words. Each line of the provided file should contain a single
	 * word followed by the elements of the corresponding word embedding
	 * (space-delimited). It is not absolutely necessary that all words in the
	 * treebank be covered by this embedding file, though the parser's
	 * performance will generally improve if you are able to provide better
	 * embeddings for more words.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;model</tt></td>
	 * <td>Yes</td>
	 * <td>Yes</td>
	 * <td>Path to a model file. If the path ends in <tt>.gz</tt>, the model
	 * will be read as a Gzipped model file. During training, we write to this
	 * path; at test time we read a pre-trained model from this path.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;textFile</tt></td>
	 * <td>No</td>
	 * <td>Yes (or <tt>testFile</tt>)</td>
	 * <td>Path to a plaintext file containing sentences to be parsed.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;testFile</tt></td>
	 * <td>No</td>
	 * <td>Yes (or <tt>textFile</tt>)</td>
	 * <td>Path to a test-set treebank in
	 * <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a> for
	 * final evaluation of the parser.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;trainFile</tt></td>
	 * <td>Yes</td>
	 * <td>No</td>
	 * <td>Path to a training treebank in
	 * <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a></td>
	 * </tr>
	 * </table>
	 *
	 * Training options:
	 * <table>
	 * <tr>
	 * <th>Option</th>
	 * <th>Default</th>
	 * <th>Description</th>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;adaAlpha</tt></td>
	 * <td>0.01</td>
	 * <td>Global learning rate for AdaGrad training</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;adaEps</tt></td>
	 * <td>1e-6</td>
	 * <td>Epsilon value added to the denominator of AdaGrad update expression
	 * for numerical stability</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;batchSize</tt></td>
	 * <td>10000</td>
	 * <td>Size of mini-batch used for training</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;clearGradientsPerIter</tt></td>
	 * <td>0</td>
	 * <td>Clear AdaGrad gradient histories every <em>n</em> iterations. If
	 * zero, no gradient clearing is performed.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;dropProb</tt></td>
	 * <td>0.5</td>
	 * <td>Dropout probability. For each training example we randomly choose
	 * some amount of units to disable in the neural network classifier. This
	 * parameter controls the proportion of units "dropped out."</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;embeddingSize</tt></td>
	 * <td>50</td>
	 * <td>Dimensionality of word embeddings provided</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;evalPerIter</tt></td>
	 * <td>100</td>
	 * <td>Run full UAS (unlabeled attachment score) evaluation every time we
	 * finish this number of iterations. (Only valid if a development treebank
	 * is provided with <tt>&#8209;devFile</tt>.)</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;hiddenSize</tt></td>
	 * <td>200</td>
	 * <td>Dimensionality of hidden layer in neural network classifier</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;initRange</tt></td>
	 * <td>0.01</td>
	 * <td>Bounds of range within which weight matrix elements should be
	 * initialized. Each element is drawn from a uniform distribution over the
	 * range <tt>[-initRange, initRange]</tt>.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;maxIter</tt></td>
	 * <td>20000</td>
	 * <td>Number of training iterations to complete before stopping and saving
	 * the final model.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;numPreComputed</tt></td>
	 * <td>100000</td>
	 * <td>The parser pre-computes hidden-layer unit activations for particular
	 * inputs words at both training and testing time in order to speed up
	 * feedforward computation in the neural network. This parameter determines
	 * how many words for which we should compute hidden-layer activations.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;regParameter</tt></td>
	 * <td>1e-8</td>
	 * <td>Regularization parameter for training</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;saveIntermediate</tt></td>
	 * <td><tt>true</tt></td>
	 * <td>If <tt>true</tt>, continually save the model version which gets the
	 * highest UAS value on the dev set. (Only valid if a development treebank
	 * is provided with <tt>&#8209;devFile</tt>.)</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;trainingThreads</tt></td>
	 * <td>1</td>
	 * <td>Number of threads to use during training. Note that depending on
	 * training batch size, it may be unwise to simply choose the maximum amount
	 * of threads for your machine. On our 16-core test machines: a batch size
	 * of 10,000 runs fastest with around 6 threads; a batch size of 100,000
	 * runs best with around 10 threads.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;wordCutOff</tt></td>
	 * <td>1</td>
	 * <td>The parser can optionally ignore rare words by simply choosing an
	 * arbitrary "unknown" feature representation for words that appear with
	 * frequency less than <em>n</em> in the corpus. This <em>n</em> is
	 * controlled by the <tt>wordCutOff</tt> parameter.</td>
	 * </tr>
	 * </table>
	 *
	 * Runtime parsing options:
	 * <table>
	 * <tr>
	 * <th>Option</th>
	 * <th>Default</th>
	 * <th>Description</th>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;escaper</tt></td>
	 * <td>N/A</td>
	 * <td>Only applicable for testing with <tt>-textFile</tt>. If provided, use
	 * this word-escaper when parsing raw sentences. (Should be a
	 * fully-qualified class name like
	 * <tt>edu.stanford.nlp.trees.international.arabic.ATBEscaper</tt>.)</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;numPreComputed</tt></td>
	 * <td>100000</td>
	 * <td>The parser pre-computes hidden-layer unit activations for particular
	 * inputs words at both training and testing time in order to speed up
	 * feedforward computation in the neural network. This parameter determines
	 * how many words for which we should compute hidden-layer activations.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;sentenceDelimiter</tt></td>
	 * <td>N/A</td>
	 * <td>Only applicable for testing with <tt>-textFile</tt>. If provided,
	 * assume that the given <tt>textFile</tt> has already been sentence-split,
	 * and that sentences are separated by this delimiter.</td>
	 * </tr>
	 * <tr>
	 * <td><tt>&#8209;tagger.model</tt></td>
	 * <td>edu/stanford/nlp/models/pos-tagger/english-left3words/english-
	 * left3words-distsim.tagger</td>
	 * <td>Only applicable for testing with <tt>-textFile</tt>. Path to a
	 * part-of-speech tagger to use to pre-tag the raw sentences before parsing.
	 * </td>
	 * </tr>
	 * </table>
	 */
	public static void main(String[] args) {
		Properties props = StringUtils.argsToProperties(args, numArgs);
		NEReconizer parser = new NEReconizer(props);

		// Train with CoNLL-X data
		if (props.containsKey("trainFile"))
			parser.train(props.getProperty("trainFile"), props.getProperty("devFile"), props.getProperty("model"),
					props.getProperty("embedFile"), props.getProperty("preModel"));

		boolean loaded = false;
		// Test with CoNLL-X data
		if (props.containsKey("testFile")) {
			parser.loadModelFile(props.getProperty("model"));
			loaded = true;
			System.err.println("##Model:"+loaded);
			parser.testCoNLL(props.getProperty("testFile"), props.getProperty("outFile"), props.getProperty("evalFile"));
		}
	}

}
