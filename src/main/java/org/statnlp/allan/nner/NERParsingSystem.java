package org.statnlp.allan.nner;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.allan.io.RAWF;

import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Defines a transition-based parsing framework for (nested) named entity recognition
 *
 * @author Allan Jie
 */
public abstract class NERParsingSystem  {

  /** A logger for this class */
  private static Redwood.RedwoodChannels log = Redwood.channels(NERParsingSystem.class);

  /**
   * Defines language-specific settings for this parsing instance.
   * Maybe not really useful for our NER experiments
   */
  @SuppressWarnings("unused")
private final TreebankLanguagePack tlp;

  /**
   * labels: entity type
   * transition: actions to do
   */
  protected List<String> labels, transitions;

  /**
   * Generate all possible transitions which this parsing system can
   * take for any given configuration.
   */
  protected abstract void makeTransitions();

  /**
   * Determine whether the given transition is legal for this
   * configuration.
   *
   * @param c Parsing configuration: stack, buffer things
   * @param t Transition string: from the transition lists
   * @return Whether the given transition is legal in this
   *         configuration
   */
  public abstract boolean canApply(NEConfiguration c, String t);

  /**
   * Apply the given transition to the given configuration, modifying
   * the configuration's state in place.
   */
  public abstract void apply(NEConfiguration c, String t);

  /**
   * Provide a static-oracle recommendation for the next parsing step
   * to take.
   *
   * @param c Current parser configuration
   * @param dSeq Gold sequence which parser needs to reach
   * @return Transition string
   */
  public abstract String getOracle(NEConfiguration c, Sequence dSeq);

//  /**
//   * Determine whether applying the given transition in the given
//   * configuration sequence will leave in us a state in which we can reach
//   * the gold sequence. (Useful for building a dynamic oracle.)
//   */
//  abstract boolean isOracle(NEConfiguration c, String t, Sequence dSeq);

  /**
   * Build an initial parser configuration from the given sentence.
   */
  public abstract NEConfiguration initialConfiguration(Sequence sentence);

  /**
   * Determine if the given configuration corresponds to a parser which
   * has completed its parse.
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

  /**
   * @param tlp TreebankLanguagePack describing the language being
   *            parsed
   * @param labels A list of possible dependency relation labels, with
   *               the ROOT relation label as the first element
   */
  public NERParsingSystem(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
    this.tlp = tlp;
    this.labels = new ArrayList<>(labels);
    //NOTE: assume that the first element of labels is rootLabel
    //in our NER case, there is no root element
    //rootLabel = labels.get(0);
    makeTransitions();

    if (verbose) {
      log.info(NEConfig.SEPARATOR);
      log.info("#Transitions: " + numTransitions());
      log.info("#Labels: " + labels.size());
      log.info("Eval_Script:"+NEConfig.EVAL_SCRIPT);
      //log.info("ROOTLABEL: " + rootLabel);
    }
  }

  /**
   * Using this one is O(n) complexity.
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
   * Evaluate performance on a list of sentences, predicted parses,
   * and gold parses.
   *
   * @return A map from metric name to metric value
   */
  public Map<String, Double> evaluate(List<Sequence> sents, List<Sequence> predictions,
                                      List<Sequence> golds, String evalOut) {
    Map<String, Double> result = new HashMap<>();

    if (predictions.size() != golds.size()) {
      log.err("Incorrect number of predictions.");
      return null;
    }
    //currently, we dun have anything to plugin to result. 
    //just show the conll eval script result
    conlleval(sents, predictions, golds, evalOut);
    //TODO: should return something, at least the f-score.
    return result;
  }

  /**
   * Evaluate the file using the evaluation script.
   * @param sents
   * @param predictions
   * @param golds
   * @param evalOut
   */
  private void conlleval(List<Sequence> sents, List<Sequence> predictions, 
		  List<Sequence> golds, String evalOut){
	  PrintWriter pw;
	  try {
		  pw = RAWF.writer(evalOut);
		  for(int pos = 0; pos < sents.size(); pos++){
			  Sequence sent = sents.get(pos);
			  Sequence prediction = predictions.get(pos);
			  Sequence gold = golds.get(pos);
			  for(int i = 0; i < sent.size(); i++){
				  pw.println(sent.get(i)[0]+" "+ gold.get(i)[0] +" "+ prediction.get(i)[0]);
			  }
			  pw.println();
		  }
		  pw.close();
	  } catch (IOException e) {
		  e.printStackTrace();
	  }
  }
  
  
	public double getAcc(List<Sequence> sentences, List<Sequence> predictions, List<Sequence> golds) {
		int corr = 0;
		int total = 0;
		for (int pos = 0; pos < sentences.size(); pos++) {
			Sequence prediction = predictions.get(pos);
			Sequence gold = golds.get(pos);
			for (int i = 0; i < gold.size(); i++) {
				if (gold.get(i)[0].equals(prediction.get(i)[0]))
					corr++;
				total++;
			}
		}
		return corr*1.0/total;
	}

}
