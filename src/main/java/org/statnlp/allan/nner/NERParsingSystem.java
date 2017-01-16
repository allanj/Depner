package org.statnlp.allan.nner;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.statnlp.allan.io.RAWF;
import org.statnlp.allan.mfl.MFLSpan;
import org.statnlp.allan.mfl.MFLUtils;

import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Defines a transition-based parsing framework for (nested) named entity
 * recognition
 *
 * @author Allan Jie
 */
public abstract class NERParsingSystem {

	/** A logger for this class */
	private static Redwood.RedwoodChannels log = Redwood.channels(NERParsingSystem.class);

	/**
	 * Defines language-specific settings for this parsing instance. Maybe not
	 * really useful for our NER experiments
	 */
	@SuppressWarnings("unused")
	private final TreebankLanguagePack tlp;

	/**
	 * labels: entity type transition: actions to do
	 */
	protected List<String> labels, transitions;

	/**
	 * Generate all possible transitions which this parsing system can take for
	 * any given configuration.
	 */
	protected abstract void makeTransitions();

	/**
	 * Determine whether the given transition is legal for this configuration.
	 *
	 * @param c
	 *            Parsing configuration: stack, buffer things
	 * @param t
	 *            Transition string: from the transition lists
	 * @return Whether the given transition is legal in this configuration
	 */
	public abstract boolean canApply(NEConfiguration c, String t);

	/**
	 * Apply the given transition to the given configuration, modifying the
	 * configuration's state in place.
	 */
	public abstract void apply(NEConfiguration c, String t);

	/**
	 * Provide a static-oracle recommendation for the next parsing step to take.
	 *
	 * @param c
	 *            Current parser configuration
	 * @param dSeq
	 *            Gold sequence which parser needs to reach
	 * @return Transition string
	 */
	public abstract String getOracle(NEConfiguration c, NEDependencyTree dTree, Sequence dner);

	// /**
	// * Determine whether applying the given transition in the given
	// * configuration sequence will leave in us a state in which we can reach
	// * the gold sequence. (Useful for building a dynamic oracle.)
	// */
	//abstract boolean isOracle(NEConfiguration c, String t, Sequence dner, DependencyTree dTree);

	/**
	 * Build an initial parser configuration from the given sentence.
	 */
	public abstract NEConfiguration initialConfiguration(Sequence sentence);

	/**
	 * Determine if the given configuration corresponds to a parser which has
	 * completed its parse.
	 */
	public abstract boolean isTerminal(NEConfiguration c);

	/**
	 * Return the number of transitions.
	 */
	public int numTransitions() {
		return transitions.size();
	}
	// TODO pass labels as Map<String, GrammaticalRelation>; use
	// GrammaticalRelation throughout

	private static HashSet<String> punct = new HashSet<>(Arrays.asList("''", ",", ".", ":", "``", "-LRB-", "-RRB-"));
	
	/**
	 * @param tlp
	 *            TreebankLanguagePack describing the language being parsed
	 * @param labels
	 *            A list of possible dependency relation labels, with the ROOT
	 *            relation label as the first element
	 */
	public NERParsingSystem(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
		this.tlp = tlp;
		this.labels = new ArrayList<>(labels);
		// NOTE: assume that the first element of labels is rootLabel
		// in our NER case, there is no root element
		// rootLabel = labels.get(0);
		makeTransitions();
		if (verbose) {
			log.info(NEConfig.SEPARATOR);
			log.info("#Transitions: " + numTransitions());
			log.info("#Labels: " + labels.size());
			log.info("Labels: " + labels.toString());
			log.info("Eval_Script:" + NEConfig.EVAL_SCRIPT);
			// log.info("ROOTLABEL: " + rootLabel);
		}
	}

	/**
	 * Using this one is O(n) complexity.
	 * 
	 * @param s
	 * @return
	 */
	public int getTransitionID(String s) {
		int numTrans = numTransitions();
		for (int k = 0; k < numTrans; ++k)
			if (transitions.get(k).equals(s))
				return k;
		return -1;
	}

	/**
	 * Evaluate performance on a list of sentences, predicted parses, and gold
	 * parses.
	 *
	 * @return A map from metric name to metric value
	 */
	public Map<String, Double> evaluate(List<Sequence> sents, List<JointPair> predictions, List<JointPair> goldPairs,
			String evalOut) {
		Map<String, Double> result = new HashMap<>();

		if (predictions.size() != goldPairs.size()) {
			log.err("Incorrect number of predictions.");
			return null;
		}
		// currently, we dun have anything to plugin to result.
		// just show the conll eval script result
		double fscore = conlleval(sents, predictions, goldPairs, evalOut);
		double uas = getUAS(sents, predictions, goldPairs)*100;
		double comb = getComb(sents, predictions, goldPairs);
		//TODO: double comb = getComb()
		result.put("fscore", fscore);
		result.put("uas", uas);
		result.put("comb", comb);
		return result;
	}

	/**
	 * Evaluate the file using the evaluation script.
	 * 
	 * @param sents
	 * @param predictions
	 * @param golds
	 * @param evalOut
	 */
	private double conlleval(List<Sequence> sents, List<JointPair> predictions, List<JointPair> goldPairs, String evalOut) {
		PrintWriter pw;
		try {
			pw = RAWF.writer(evalOut);
			for (int pos = 0; pos < sents.size(); pos++) {
				Sequence sent = sents.get(pos);
				Sequence nerPrediction = predictions.get(pos).ners;
				Sequence gold = goldPairs.get(pos).ners;
				for (int i = 1; i < sent.size(); i++) {
					String pred = nerPrediction.tokens[i].ner();
					if (pred.startsWith("S"))
						pred = "B" + pred.substring(1);
					if (pred.startsWith("E"))
						pred = "I" + pred.substring(1);
					pw.println(sent.tokens[i].word() + " " + sent.tokens[i].tag() + " " + gold.tokens[i].ner() + " " + pred);
				}
				pw.println();
			}
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return evalNER(evalOut);
	}
	
	/**
	 * Evaluate the file using the conlleval{@link http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt } script.
	 * @param outputFile
	 * @throws IOException
	 */
	private double evalNER(String outputFile){
		double fscore = Double.NEGATIVE_INFINITY;
		try{
			System.out.println("perl data/semeval10t1/conlleval.pl < "+outputFile);
			ProcessBuilder pb = null;
			if(NEConfig.OS.equals("windows")){
				pb = new ProcessBuilder("D:/Perl64/bin/perl","E:/Framework/data/semeval10t1/conlleval.pl"); 
			}else{
				pb = new ProcessBuilder(NEConfig.EVAL_SCRIPT); 
			}
			pb.redirectInput(new File(outputFile));
			//pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			//pb.redirectError(ProcessBuilder.Redirect.INHERIT);
			Process process = pb.start();
			BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			while ((line = in.readLine()) != null ) {
				System.out.println(line);
				if (line.startsWith("accuracy")){
					String[] vals = line.split("\\s+");
					fscore = Double.parseDouble(vals[vals.length-1]);
				}
			}
			process.waitFor();
			in.close();
		}catch(IOException ioe){
			ioe.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return fscore;
	}
	
	private double getUAS(List<Sequence> sents, List<JointPair> predictions, List<JointPair> goldPairs) {
		int total = 0;
		int corr = 0;
		for (int pos = 0; pos < sents.size(); pos++) {
			Sequence sent = sents.get(pos);
			NEDependencyTree preDep = predictions.get(pos).tree;
			NEDependencyTree gold = goldPairs.get(pos).tree;
			for (int i = 1; i < sent.size(); ++i) {
				if (!punct.contains(sent.tokens[i].tag())) {
					if (preDep.getHead(i) == gold.getHead(i))
						corr++;
					total++;
				}
			}
		}
		System.out.printf("UAS: %.2f\n", corr*1.0/total*100);
		return corr*1.0/total;
	}
	
	private double getComb(List<Sequence> sents, List<JointPair> predictions, List<JointPair> goldPairs) {
		int tp = 0;
		int tp_fp = 0;
		int tp_fn = 0;
		for(int index = 0; index < sents.size(); index++){
			Sequence sent = sents.get(index);
			List<MFLSpan> outputSpans = MFLUtils.toSpan(goldPairs.get(index).ners, goldPairs.get(index).tree);
			List<MFLSpan> predSpans = MFLUtils.toSpan(predictions.get(index).ners, predictions.get(index).tree);
			Map<MFLSpan, MFLSpan> outputMap = new HashMap<MFLSpan, MFLSpan>(outputSpans.size());
			for (MFLSpan outputSpan : outputSpans) outputMap.put(outputSpan, outputSpan);
			Map<MFLSpan, MFLSpan> predMap = new HashMap<MFLSpan, MFLSpan>(outputSpans.size());
			for (MFLSpan predSpan : predSpans) predMap.put(predSpan, predSpan);
			for (MFLSpan predSpan: predSpans) {
				if (predSpan.start == predSpan.end && punct.contains(sent.tokens[predSpan.start].tag())) {
					continue;
				}
				if (predSpan.start == predSpan.end && predSpan.start == 0) {
					continue;
				}
				if (outputMap.containsKey(predSpan)) {
					MFLSpan outputSpan = outputMap.get(predSpan);
					Set<Integer> intersection = new HashSet<Integer>(predSpan.heads);
					intersection.retainAll(outputSpan.heads);
					tp += intersection.size();
				}
				tp_fp += predSpan.heads.size();
			}
			
			for (MFLSpan outputSpan: outputSpans) {
				if (outputSpan.start == outputSpan.end && punct.contains(sent.tokens[outputSpan.start].tag())) {
					continue;
				}
				if (outputSpan.start == outputSpan.end && outputSpan.start == 0) {
					continue;
				}
				tp_fn += outputSpan.heads.size();
			}
		}
		double precision = tp*1.0 / tp_fp * 100;
		double recall = tp*1.0 / tp_fn * 100;
		double fmeasure = 2.0*tp / (tp_fp + tp_fn) * 100;
		System.out.printf("[Unit Attachment Evaluation]\n");
		System.out.printf("TP: %d, TP+FP: %d, TP+FN: %d\n", tp, tp_fp, tp_fn);
		System.out.printf("Precision: %.2f%%, Recall: %.2f%%, F-measure: %.2f%%\n", precision, recall, fmeasure);
		return fmeasure;
	}

}
