
package org.statnlp.allan.nner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.logging.Redwood;

/**
 *
 * Some utility functions
 *
 * @author Danqi Chen
 * @author Jon Gauthier
 */

public class NEUtil {

	/** A logger for this class */
	private static Redwood.RedwoodChannels log = Redwood.channels(NEUtil.class);

	private NEUtil() {
	} // static methods

	private static Random random;

	/**
	 * Normalize word embeddings by setting mean = rMean, std = rStd
	 */
	public static double[][] scaling(double[][] A, double rMean, double rStd) {
		int count = 0;
		double mean = 0.0;
		double std = 0.0;
		for (double[] aA : A)
			for (int j = 0; j < aA.length; ++j) {
				count += 1;
				mean += aA[j];
				std += aA[j] * aA[j];
			}
		mean = mean / count;
		std = Math.sqrt(std / count - mean * mean);

		System.err.printf("Scaling word embeddings:");
		System.err.printf("(mean = %.2f, std = %.2f) -> (mean = %.2f, std = %.2f)", mean, std, rMean, rStd);

		double[][] rA = new double[A.length][A[0].length];
		for (int i = 0; i < rA.length; ++i)
			for (int j = 0; j < rA[i].length; ++j)
				rA[i][j] = (A[i][j] - mean) * rStd / std + rMean;
		return rA;
	}

	/**
	 * Normalize word embeddings by setting mean = 0, std = 1
	 */
	public static double[][] scaling(double[][] A) {
		return scaling(A, 0.0, 1.0);
	}

	// return strings sorted by frequency, and filter out those with freq. less
	// than cutOff.

	/**
	 * Build a dictionary of words collected from a corpus.
	 * <p>
	 * Filters out words with a frequency below the given {@code cutOff}.
	 *
	 * @return Words sorted by decreasing frequency, filtered to remove any
	 *         words with a frequency below {@code cutOff}
	 */
	public static List<String> generateDict(List<String> str, int cutOff) {
		Counter<String> freq = new IntCounter<>();
		for (String aStr : str)
			freq.incrementCount(aStr);

		List<String> keys = Counters.toSortedList(freq, false);
		List<String> dict = new ArrayList<>();
		for (String word : keys) {
			if (freq.getCount(word) >= cutOff)
				dict.add(word);
		}
		return dict;
	}

	public static List<String> generateDict(List<String> str) {
		return generateDict(str, 1);
	}

	/**
	 * @return Shared random generator used in this package
	 */
	static Random getRandom() {
		if (random != null)
			return random;
		else
			return getRandom(System.currentTimeMillis());
	}

	/**
	 * Set up shared random generator to use the given seed.
	 *
	 * @return Shared random generator object
	 */
	static Random getRandom(long seed) {
		random = new Random(seed);
		System.err.printf("Random generator initialized with seed %d%n", seed);

		return random;
	}

	public static <T> List<T> getRandomSubList(List<T> input, int subsetSize) {
		int inputSize = input.size();
		if (subsetSize > inputSize)
			subsetSize = inputSize;

		Random random = getRandom();
		for (int i = 0; i < subsetSize; i++) {
			int indexToSwap = i + random.nextInt(inputSize - i);
			T temp = input.get(i);
			input.set(i, input.get(indexToSwap));
			input.set(indexToSwap, temp);
		}
		return input.subList(0, subsetSize);
	}

	public static void loadConllFile(String inFile, List<Sequence> sents, List<Sequence> ners, boolean unlabeled,
			boolean cPOS) {

		BufferedReader reader = null;
		try {
			reader = IOUtils.readerFromString(inFile);

			List<CoreLabel> sent = new ArrayList<>();
			List<CoreLabel> nerSeq = new ArrayList<>();

			for (String line : IOUtils.getLineIterable(reader, false)) {
				String[] splits = line.split("\t");
				if (splits.length < 10) {
					if (sent.size() > 0) {
						CoreLabel[] nerArr = new CoreLabel[nerSeq.size()];
						nerSeq.toArray(nerArr);
						ners.add(new NESeq(nerArr));
						CoreLabel[] sentArr = new CoreLabel[sent.size()];
						sent.toArray(sentArr);
						sents.add(new Sent(sentArr));
						sent = new ArrayList<>();
						nerSeq = new ArrayList<>();
					}
				} else {
					String word = splits[1], pos = cPOS ? splits[3] : splits[4], label = splits[10];

					CoreLabel token = new CoreLabel();
					token.setWord(word);
					token.setTag(pos);
					sent.add(token);

					CoreLabel nerToken = new CoreLabel();
					if (!unlabeled) {
						nerToken.setNER(label);
						nerSeq.add(nerToken);
					} else {
						nerToken.setNER(NEConfig.UNKNOWN); // unknown ne type
						nerSeq.add(nerToken);
					}
				}
			}
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		} finally {
			IOUtils.closeIgnoringExceptions(reader);
		}
	}

	public static void loadConllFile(String inFile, List<Sequence> sents, List<Sequence> ners) {
		loadConllFile(inFile, sents, ners, false, false);
	}

	public static void writeConllFile(String outFile, List<Sequence> sentences, List<Sequence> ners) {
		try {
			PrintWriter output = IOUtils.getPrintWriter(outFile);

			for (int i = 0; i < sentences.size(); i++) {
				Sequence sentence = sentences.get(i);
				Sequence ner = ners.get(i);

				for (int j = 0; j < sentence.size(); ++j) {
					String word = sentence.get(j)[0];
					String tag = sentence.get(j)[1];
					output.printf("%d\t%s\t_\t%s\t%s\t_\t%s\t_\t_%n", j, word, tag, tag, ner.get(j)[0]);
				}
				output.println();
			}
			output.close();
		} catch (Exception e) {
			throw new RuntimeIOException(e);
		}
	}

	public static void printNERStats(String str, List<Sequence> ners) {
		log.info(NEConfig.SEPARATOR + " " + str);
		int nNER = ners.size();
		System.err.printf("#NER sents: %d%n", nNER);
	}

	public static void printTreeStats(List<Sequence> ners) {
		printNERStats("", ners);
	}
}
