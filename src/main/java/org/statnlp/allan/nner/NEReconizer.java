package org.statnlp.allan.nner;

import static java.util.stream.Collectors.toList;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.process.WordShapeClassifier;
import edu.stanford.nlp.util.RuntimeInterruptedException;
import edu.stanford.nlp.util.StringUtils;
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

	private Map<String, Integer> feature2Idx;

	/**
	 * Given a particular recongnizer configuration, this classifier will
	 * predict the best transition to make next.
	 *
	 * The {@link edu.stanford.nlp.parser.nndep.Classifier} class handles both
	 * training and inference.
	 */
	private StructuredPerceptron classifier;
	private NERParsingSystem system;

	private final NEConfig config;
	private boolean finalized;

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


	private int toFeature(String str) {
		if (this.feature2Idx.containsKey(str))
			return this.feature2Idx.get(str);
		else {
			if (finalized) return -1;
			int idx = this.feature2Idx.size();
			this.feature2Idx.put(str, idx);
			return idx;
		}
	}
	
	
	public List<Integer> getFeatures(NEConfiguration c, String action) {
		// Presize the arrays for very slight speed gain. Hardcoded, but so is
		// the current feature list.
		// Not specify the initial capacity will make the list dynamically
		// resize
		List<Integer> feature = new ArrayList<>(44);
		int s0 = c.getStack(0);
		String s0w = c.getWord(s0);
		String s0p = c.getPOS(s0);
		int s1 = c.getStack(1);
		String s1w = c.getWord(s1);
		String s1p = c.getPOS(s1);
		int b0 = c.getBuffer(0);
		int b1 = c.getBuffer(1);
		int b2 = c.getBuffer(2);
		String b0w = c.getWord(b0);
		String b0p = c.getPOS(b0);
		String b1w = c.getWord(b1);
		String b1p = c.getPOS(b1);
		String b2w = c.getWord(b2);
		String b2p = c.getPOS(b2);
//		log.info(c.getStr());
//		log.info(s0w + " || " + b0w + " " + b1w + " " + b2w + " ACTION:"+action);
//		log.info(s0p + " || " + b0p + " " + b1p + " " + b2p + " ACTION:"+action);
		feature.add(toFeature("S0wp=" + s0w + " & " + s0p + " LABELS:" + action));
		feature.add(toFeature("S0w=" + s0w + " LABELS:" + action));
		feature.add(toFeature("S0p=" + s0p + " LABELS:" + action));
		feature.add(toFeature("S1wp=" + s1w + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S1w=" + s1w + " LABELS:" + action));
		feature.add(toFeature("S1p=" + s1p + " LABELS:" + action));
		feature.add(toFeature("B0wp=" + b0w + " & " + b0p + " LABELS:" + action));
		feature.add(toFeature("B0w=" + b0w + " LABELS:" + action));
		feature.add(toFeature("B0p=" + b0p + " LABELS:" + action));
		feature.add(toFeature("B1wp=" + b1w + " & " + b1p + " LABELS:" + action));
		feature.add(toFeature("B1w=" + b1w + " LABELS:" + action));
		feature.add(toFeature("B1p=" + b1p + " LABELS:" + action));
//		feature.add(toFeature("B2wp=" + b2w + " & " + b2p + " LABELS:" + action));
//		feature.add(toFeature("B2w=" + b2w + " LABELS:" + action));
//		feature.add(toFeature("B2p=" + b2p + " LABELS:" + action));
		
		feature.add(toFeature("S0wpS1wp=" + s0w + " & " + s0p + "&" + s1w + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0wpS1w=" + s0w + " & " + s0p + "&" + s1w + " LABELS:" + action));
		feature.add(toFeature("S0wpS1p=" + s0w + " & " + s0p + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0wS1wp=" + s0w + "&" + s1w + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0pS1wp=" + s0p + "&" + s1w + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0wS1w=" + s0w + "&" + s1w + " LABELS:" + action));
		feature.add(toFeature("S0pS1p=" + s0p + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0pB0p=" + s0p + "&" + b0p + " LABELS:" + action));
		
		int dist = s0 - s1;
		feature.add(toFeature("S1wd=" + s1w + " & " + dist + " LABELS:" + action));
		feature.add(toFeature("S1pd=" + s1p + " & " + dist + " LABELS:" + action));
		feature.add(toFeature("S0wd=" + s0w + " & " + dist + " LABELS:" + action));
		feature.add(toFeature("S0pd=" + s0p + " & " + dist + " LABELS:" + action));
		feature.add(toFeature("S1wS0wd=" + s1w + " & " + s0w + " & " + dist + " LABELS:" + action));
		feature.add(toFeature("S1pS0pd=" + s1p + " & " + s0p + " & " + dist + " LABELS:" + action));
		
		//valency
		int s1vl = c.getLeftValency(s1);
		int s1vr = c.getRightValency(s1);
		int s0vl = c.getLeftValency(s0);
		int s0vr = c.getRightValency(s0);
		feature.add(toFeature("S1wvr=" + s0w + " & " + s1vr + " LABELS:" + action));
		feature.add(toFeature("S1pvr=" + s0p + " & " + s1vr + " LABELS:" + action));
		feature.add(toFeature("S1wvl=" + s0w + " & " + s1vl + " LABELS:" + action));
		feature.add(toFeature("S1pvl=" + s0p + " & " + s1vl + " LABELS:" + action));
		feature.add(toFeature("S0wvl=" + s0w + " & " + s0vl + " LABELS:" + action));
		feature.add(toFeature("S0pvl=" + s0p + " & " + s0vl + " LABELS:" + action));
		feature.add(toFeature("S0wvr=" + s0w + " & " + s0vr + " LABELS:" + action));
		feature.add(toFeature("S0pvr=" + s0p + " & " + s0vr + " LABELS:" + action));
		
		int s0l = c.getLeftChild(s0);
		int s0r = c.getRightChild(s0);
		String s0lw = c.getWord(s0l);
		String s0lp = c.getPOS(s0l);
		String s0rw = c.getWord(s0r);
		String s0rp = c.getPOS(s0r);
		int s1l = c.getLeftChild(s1);
		int s1r = c.getRightChild(s1);
		String s1lw = c.getWord(s1l);
		String s1lp = c.getPOS(s1l);
		String s1rw = c.getWord(s1r);
		String s1rp = c.getPOS(s1r);
		feature.add(toFeature("S1lw=" + s1lw + " LABELS:" + action));
		feature.add(toFeature("S1lp=" + s1lp + " LABELS:" + action));
		feature.add(toFeature("S1rw=" + s1rw + " LABELS:" + action));
		feature.add(toFeature("S1rp=" + s1rp + " LABELS:" + action));
		feature.add(toFeature("S0lw=" + s0lw + " LABELS:" + action));
		feature.add(toFeature("S0lp=" + s0lp + " LABELS:" + action));
		feature.add(toFeature("S0rw=" + s0rw + " LABELS:" + action));
		feature.add(toFeature("S0rp=" + s0rp + " LABELS:" + action));
		
		int s0l2 = c.getLeftChild(s0, 2);
		int s0r2 = c.getRightChild(s0, 2);
		int s1l2 = c.getLeftChild(s1, 2);
		int s1r2 = c.getRightChild(s1, 2);
		String s0l2w = c.getWord(s0l2);
		String s0l2p = c.getPOS(s0l2);
		String s0r2w = c.getWord(s0r2);
		String s0r2p = c.getPOS(s0r2);
		String s1l2w = c.getWord(s1l2);
		String s1l2p = c.getPOS(s1l2);
		String s1r2w = c.getWord(s1r2);
		String s1r2p = c.getPOS(s1r2);
		feature.add(toFeature("S1l2w=" + s1l2w + " LABELS:" + action));
		feature.add(toFeature("S1l2p=" + s1l2p + " LABELS:" + action));
		feature.add(toFeature("S1r2w=" + s1r2w + " LABELS:" + action));
		feature.add(toFeature("S1r2w=" + s1r2p + " LABELS:" + action));
		feature.add(toFeature("S0l2w=" + s0l2w + " LABELS:" + action));
		feature.add(toFeature("S0l2p=" + s0l2p + " LABELS:" + action));
		feature.add(toFeature("S0r2w=" + s0r2w + " LABELS:" + action));
		feature.add(toFeature("S0r2w=" + s0r2p + " LABELS:" + action));
		feature.add(toFeature("S1pS1lpS1l2p=" + s1p + " & " + s1lp + " & " + s1l2p + " LABELS:" + action));
		feature.add(toFeature("S1pS1rpS1r2p=" + s1p + " & " + s1rp + " & " + s1r2p + " LABELS:" + action));
		feature.add(toFeature("S0pS0lpS0l2p=" + s0p + " & " + s0lp + " & " + s0l2p + " LABELS:" + action));
		feature.add(toFeature("S0pS0rpS0r2p=" + s0p + " & " + s0rp + " & " + s0r2p + " LABELS:" + action));
		
		//three word feature
		feature.add(toFeature("S1pS0pB0p=" + s1p + " & " + s0p + " & " + b0p + " LABELS:" + action));
		feature.add(toFeature("S1pS0pS0lp=" + s1p + " & " + s0p + " & " + s0lp + " LABELS:" + action));
		feature.add(toFeature("S1pS0pS0rp=" + s1p + " & " + s0p + " & " + s0rp + " LABELS:" + action));
		feature.add(toFeature("S1pS0pS1lp=" + s1p + " & " + s0p + " & " + s1lp + " LABELS:" + action));
		feature.add(toFeature("S1pS0pS1rp=" + s1p + " & " + s0p + " & " + s1rp + " LABELS:" + action));
		feature.add(toFeature("S1pS0wS1rp=" + s1p + " & " + s0w + " & " + s1rp + " LABELS:" + action));
		feature.add(toFeature("S1pS0wS1lp=" + s1p + " & " + s0w + " & " + s1lp + " LABELS:" + action));
		feature.add(toFeature("S1pS0wB0p=" + s1p + " & " + s0w + " & " + b0p + " LABELS:" + action));
		
		
		String b0s = WordShapeClassifier.wordShape(b0w,WordShapeClassifier.WORDSHAPEJENNY1);
		feature.add(toFeature("E-B0s=" + b0s + " LABELS:" + action));
		String s0ner = c.getLabel(s0);
		String s1ner = c.getLabel(s1);
		feature.add(toFeature("S0ner=" + s0ner + " LABELS:" + action));
		feature.add(toFeature("S1ner=" + s1ner + " LABELS:" + action));
		feature.add(toFeature("S0nerS1ner=" + s0ner + " & " + s1ner + " LABELS:" + action));
		feature.add(toFeature("S0nerS0wS1nerS1w=" + s0ner + "&" + s0w + " & " + s1ner + " & " + s1w + " LABELS:" + action));
		feature.add(toFeature("S0nerS0pS1nerS1p=" + s0ner + "&" + s0p + " & " + s1ner + " & " + s1p + " LABELS:" + action));
		feature.add(toFeature("S0nerS0wS1nerS1wB0w=" + s0ner + "&" + s0w + " & " + s1ner + " & " + s1w + " & " + b0w + " LABELS:" + action));
		feature.add(toFeature("S0nerS0pS1nerS1pB0p=" + s0ner + "&" + s0p + " & " + s1ner + " & " + s1p + " & " + b0p + " LABELS:" + action));
		feature.add(toFeature("S0wS0ner=" + s0w + " & " + s0ner + " LABELS:" + action));
		feature.add(toFeature("S0pS0ner=" + s0p + " & " + s0ner + " LABELS:" + action));
		feature.add(toFeature("S0wS0pS0ner=" + s0w + " & " + s0p + " & " + s0ner + " LABELS:" + action));
		feature.add(toFeature("S0nerB0w=" + s0ner + " & " + b0w + " LABELS:" + action));
		feature.add(toFeature("S0nerB0p=" + s0ner + " & " + b0p + " LABELS:" + action));
		int bMinus1 = b0 - 1;
		String bMinus1w = c.getWord(bMinus1);
		String bMius1p = c.getPOS(bMinus1);
		String bMinus1s = WordShapeClassifier.wordShape(bMinus1w,WordShapeClassifier.WORDSHAPEJENNY1);
				
		String bMinus1NE = c.getLabel(bMinus1);
		String bMinus1p = c.getPOS(bMinus1);
		feature.add(toFeature("B-1w=" + bMinus1w + " LABELS:" + action));
		feature.add(toFeature("B-1p=" + bMius1p + " LABELS:" + action));
		feature.add(toFeature("B-1wB-1ner=" + bMinus1w + " & " + bMinus1NE + " LABELS:" + action));
		feature.add(toFeature("B-1pB-1ner=" + bMius1p  + " & " + bMinus1NE + " LABELS:" + action));
		feature.add(toFeature("E-B-1s=" + bMinus1s + " LABELS:" + action));
		String b1s = WordShapeClassifier.wordShape(b1w,WordShapeClassifier.WORDSHAPEJENNY1);
		feature.add(toFeature("B1s=" + b1s + " LABELS:" + action));
		feature.add(toFeature("B0-surrp=" + bMinus1p + " & " + b1p + " LABELS:" + action));
		feature.add(toFeature("B0-surrs=" + bMinus1s + " & " + b1s + " LABELS:" + action));
		int bMinus2 = b0 - 2;
		int bMinus3 = b0 - 3;
		int bMinus4 = b0 - 4;
		int b3 = c.getBuffer(3);
		int b4 = c.getBuffer(4);
		String bMinus2w = c.getWord(bMinus2);
		String bMinus3w = c.getWord(bMinus3);
		String bMinus4w = c.getWord(bMinus4);
		String b3w = c.getWord(b3);
		String b4w = c.getWord(b4);
		feature.add(toFeature("E-left4=" + bMinus1w + " & " + bMinus2w + " & " + bMinus3w + " & " + bMinus4w + " LABELS:" + action));
		feature.add(toFeature("E-right4=" + b1w + " & " + b2w + " & " + b3w + " & " + b4w + " LABELS:" + action));
		
		for (int l = 1; l <= b0w.length(); l++) {
			for (int sp = 0; sp <= b0w.length() - l; sp++) {
				feature.add(toFeature("E-ngram-" + l + "=" + b0w.substring(sp, sp + l) + " LABELS:" + action));
			}
		}
		
		for(int plen = 1; plen <= 3; plen++){
			if(b0w.length() >= plen){
				String suff = b0w.substring(b0w.length()-plen, b0w.length());
				feature.add(toFeature("E-suff=" + suff + " & " + plen + " LABELS:" + action));
				String pref = b0w.substring(0,plen);
				feature.add(toFeature("E-pref=" + pref + " & " + plen + " LABELS:" + action));
			}
		}
		feature.add(toFeature("transition=" + bMinus1NE + " LABELS:" + action));
		return feature;
	}

	/**
	 * Extracting features and also generate the training samples
	 * @param sents
	 * @param ners
	 * @param trees
	 * @return
	 */
	public NEDataset genTrainExamples(List<Sequence> sents, List<JointPair> pairs) {
		int numTrans = system.numTransitions();
		NEDataset ret = new NEDataset(sents.size(), numTrans);

		log.info(NEConfig.SEPARATOR);
		//log.info("Generate training examples...");
		log.info("Extracting features from training data...");
		this.feature2Idx = new HashMap<String, Integer>();
		for (int i = 0; i < sents.size(); ++i) {
			//log.info("sent "+i+": size: " + sents.get(i).size());
			//log.info("sent "+i+": size: " + sents.get(i).toString());
			NEConfiguration c = system.initialConfiguration(sents.get(i));
			List<List<Integer>> goldFeatures = new ArrayList<List<Integer>>();
			List<String> goldActionSeq = new ArrayList<String>();
			JointPair pair = pairs.get(i);
			//log.info(pair.tree.head.toString());
			if (pair.tree.isProjective()) {
				while (!system.isTerminal(c)) {
					//log.info(c.getStr());
					String oracle = system.getOracle(c, pair.tree, pair.ners);
					List<Integer> goldFeature = getFeatures(c, oracle);
					goldFeatures.add(goldFeature);
					goldActionSeq.add(oracle);
//					for (int j = 0; j < numTrans; ++j) {
//						String action = system.transitions.get(j); 
////						List<Integer> feature = getFeatures(c, action);
//						getFeatures(c, action);
//					}
					//log.info("apply: " + oracle);
					system.apply(c, oracle);
				}
				ret.addExample(goldFeatures, goldActionSeq);
			}
		}
		log.info("#Train Examples: " + ret.n);
		//assert(ret.n == sents.size());
		
		return ret;
	}


	private List<String> generateNERLabels(List<JointPair> trainPairs) {
		Set<String> set = new HashSet<String>();
		for (JointPair pair: trainPairs) {
			for (int i = 0; i < pair.ners.size(); i++) {
				set.add(pair.ners.tokens[i].ner());
			}
		}
		return new ArrayList<String>(set);
	}
	
	public void writeModelFile(String modelFile) {
		try {
			double[] weights = classifier.getWeights();
			Writer output = IOUtils.getPrintWriter(modelFile);
			// Now write classifier weights
			for (int i = 0; i < weights.length; ++i) {
				output.write(String.valueOf(weights[i]));
				if (i == weights.length - 1)
					output.write("\n");
				else
					output.write(" ");
			}
			output.close();
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}
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
	 */
	public void train(String trainFile, String devFile, String modelFile) {
		log.info("Train File: " + trainFile);
		log.info("Dev File: " + devFile);
		log.info("Model File: " + modelFile);

		List<Sequence> trainSents = new ArrayList<>();
		List<JointPair> trainPairs = new ArrayList<>();
		
		NEUtil.loadConllFile(trainFile, trainSents, trainPairs, config.IOBESencoding);
		NEUtil.printNERStats("Train", trainPairs);
		NEUtil.printTreeStats("Train", trainPairs);

		List<Sequence> devSents = new ArrayList<>();
		List<JointPair> devPairs = new ArrayList<>();
		if (devFile != null) {
			NEUtil.loadConllFile(devFile, devSents, devPairs, false);
			NEUtil.printNERStats("Dev", devPairs);
			NEUtil.printTreeStats("Dev", devPairs);
		}
		List<String> lDict = generateNERLabels(trainPairs);
		system = new NEStandard(config.tlp, lDict, true, config.IOBESencoding);

		// Initialize a classifier; prepare for training
		setupClassifierForTraining(trainSents, trainPairs);
		this.classifier.enableAvgPerceptron();

		log.info(NEConfig.SEPARATOR);
		config.printParameters();
		int numTrans = system.numTransitions();
		long startTime = System.currentTimeMillis();
		/**
		 * Track the best accracy performance we've seen.
		 */
		double bestFscore = 0;
		
		for (int iter = 0; iter < config.maxIter; ++iter) {
			log.info("##### Iteration " + iter);
			int correctCount = 0;
			int total = 0;
			int n = 0;
			for (int idx = 0; idx < trainSents.size(); idx++) {
				if (!trainPairs.get(idx).tree.isProjective()) continue;
				Sequence sent = trainSents.get(idx);
				NEConfiguration c = system.initialConfiguration(sent);
				List<String> goldActSeq = classifier.trainingData.examples.get(n).getActionSequences();
				//System.err.println(goldActSeq.toString());
				int gIdx = 0;
				while (!system.isTerminal(c)) {
					double maxScore = Double.NEGATIVE_INFINITY;
					String bestAct = null;
					List<Integer> bestFeature = null;
					//log.info(c.getStr());
					for (int j = 0; j < numTrans; ++j) {
						String action = system.transitions.get(j);
						//log.info("candidate act:"  + action);
						if (system.canApply(c, action)) {
							List<Integer> feature = getFeatures(c, action);
							double score = classifier.getScore(feature, true);
							//log.info("candidate act:"  + action + " score: " + score);
							if (score > maxScore) {
								bestAct = action;
								maxScore = score;
								bestFeature = feature;
							}
						}
					}
					if (bestAct == null) throw new RuntimeException("No action is available??");
					if (!bestAct.equals(goldActSeq.get(gIdx))) {
						//update parameter
						List<Integer> goldFeature = classifier.trainingData.examples.get(n).getFeatures().get(gIdx);
						this.classifier.update(goldFeature, bestFeature);
						this.classifier.incrementAvgWeight();
						break;
					} else {
						system.apply(c, bestAct);
						this.classifier.incrementAvgWeight();
						gIdx++;
					}
				}
				n++;
				if (system.isTerminal(c))
					correctCount++;
				total++;
			}
			System.err.printf(" Correct(%%) = %.2f%%\n", correctCount*1.0/total*100);
			log.info("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");

			// Fscore evaluation
			if (devFile != null && iter % config.evalPerIter == 0) {
				this.classifier.averageWeight();
				List<JointPair> predicted = devSents.stream().map(this::predictInner).collect(toList());
				Map<String, Double> result = system.evaluate(devSents, predicted, devPairs, NEConfig.EVAL_FILE);
				double fscore = result.get("fscore");   
				double uas = result.get("uas");
				double comb = result.get("comb");
				if (config.saveIntermediate && comb > bestFscore) {
					System.err.printf("Exceeds best previous comb fscore of %.2f. Saving model file..%n", bestFscore);
					bestFscore = comb;
					writeModelFile(modelFile);
				}
			}
		}
		this.classifier.averageWeight();
		this.classifier.finalizeClassifier();
		if (devFile != null) {
			// Do final UAS evaluation and save if final model beats the
			// best intermediate one
			List<JointPair> predicted = devSents.stream().map(this::predictInner).collect(toList());
			Map<String, Double> result = system.evaluate(devSents, predicted, devPairs, NEConfig.EVAL_FILE);
			double fscore = result.get("fscore");   
			double uas = result.get("uas");
			double comb = result.get("comb");
			if (comb > bestFscore) {
				System.err.printf("Final model F-score: %f%n", comb);
				System.err.printf("Exceeds best previous comb fscore of %.2f. Saving model file..%n", bestFscore);
				writeModelFile(modelFile);
			}
		} else {
			writeModelFile(modelFile);
		}
	}


	/**
	 * Prepare a classifier for training with the given dataset.
	 */
	private void setupClassifierForTraining(List<Sequence> trainSents, List<JointPair> trainPairs) {
		
		// Randomly initialize weight matrices
		this.finalized = false;
		NEDataset trainSet = genTrainExamples(trainSents, trainPairs);
		this.finalized = true;
		log.info("Total number of features: " + this.feature2Idx.size());
		double[] weights = new double[this.feature2Idx.size()];
//		Random random = NEUtil.getRandom();
//	    for (int i = 0; i < weights.length; i++)
//	        weights[i] = random.nextDouble() * 2 * config.initRange - config.initRange;
		//initialization:
		classifier = new StructuredPerceptron(config, weights, system.numTransitions(), trainSet);
	}

	/**
	 * Determine the named entity sequence given the input sequence
	 * <p>
	 * This "inner" method returns a structure unique to this package; 
	 */
	private JointPair predictInner(Sequence sentence) {
		int numTrans = system.numTransitions();

		NEConfiguration c = system.initialConfiguration(sentence);
		while (!system.isTerminal(c)) {
			if (Thread.interrupted()) { // Allow interrupting
				throw new RuntimeInterruptedException();
			}
			//double[] scores = classifier.computeScores(getFeatureArray(c));

			double optScore = Double.NEGATIVE_INFINITY;
			String optTrans = null;

			for (int j = 0; j < numTrans; ++j) {
				String action = system.transitions.get(j);
				List<Integer> featureList =  getFeatures(c, action);
				double score = classifier.getScore(featureList, false);
				if (score > optScore && system.canApply(c, system.transitions.get(j))) {
					optScore = score;
					optTrans = action;
				}
			}
			system.apply(c, optTrans);
		}
		return new JointPair(c.ners, c.tree);
	}

	/**
	 * Determine the dependency parse of the given sentence using the loaded
	 * model. You must first load a parser before calling this method.
	 *
	 * @throws java.lang.IllegalStateException
	 *             If parser has not yet been loaded and initialized (see
	 *             {@link #initialize(boolean)}
	 */
	public JointPair predict(Sequence sentence) {
		if (system == null)
			throw new IllegalStateException("Parser has not been  " + "loaded and initialized; first load a model.");
		JointPair result = predictInner(sentence);
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
	public double testCoNLL(String testFile, String outFile) {
		log.info("Test File: " + testFile);
		List<Sequence> testSents = new ArrayList<>();
		List<JointPair> testPairs = new ArrayList<>();
		NEUtil.loadConllFile(testFile, testSents, testPairs, false); //no iobes encoding when we test the file

		List<JointPair> predicted = testSents.stream().map(this::predictInner).collect(toList()); // jdk 8 new features
		Map<String, Double> result = system.evaluate(testSents, predicted, testPairs, NEConfig.EVAL_FILE);

		double fscore = result.get("fscore");
		System.err.printf("F-score = %.2f%n", fscore);

		if (outFile != null) {
			NEUtil.writeConllFile(outFile, testSents, predicted);
		}
		return fscore;
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
	 * <tr>
	 * <td><tt>&#8209;iobes</tt></td>
	 * <td>truer</td>
	 * <td>IOBES encoding in NER
	 * </td>
	 * </tr>
	 * </table>
	 */
	public static void main(String[] args) {
		Properties props = StringUtils.argsToProperties(args, numArgs);
		NEReconizer parser = new NEReconizer(props);
		NEConfig.OS = props.getProperty("os").equals("windows")? "windows":"mac";
		// Train with CoNLL-X data
		if (props.containsKey("trainFile"))
			parser.train(props.getProperty("trainFile"), props.getProperty("devFile"), props.getProperty("model"));

//		boolean loaded = false;
//		// Test with CoNLL-X data
//		if (props.containsKey("testFile")) {
//			parser.loadModelFile(props.getProperty("model"));
//			loaded = true;
//			System.err.println("##Model:"+loaded);
//			parser.testCoNLL(props.getProperty("testFile"), props.getProperty("outFile"));
//		}
	}

}
