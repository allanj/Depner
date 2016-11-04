package org.statnlp.allan.nner;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.trees.TreebankLanguagePack;


/**
 * Define an NE standard transition-based parser for named entity recognition 
 * A very basic transition-based approach for named entity recognition.
 *
 * @author Allan Jie
 */
public class NEStandard extends NERParsingSystem {

  public NEStandard(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
    super(tlp, labels, verbose);
  }

  
  @Override
  public boolean isTerminal(NEConfiguration c) {
	//means all words are shifted
    return c.getBufferSize() == 0;
  }

  @Override
  public void makeTransitions() {
    transitions = new ArrayList<>();
    //only have shift operation
    for (String label : labels)
      transitions.add("S(" + label + ")");
    
  }

  @Override
  public NEConfiguration initialConfiguration(Sequence s) {
    NEConfiguration c = new NEConfiguration(s);
    int length = s.size();

    // For each token, add dummy elements to the configuration's tree
    // and add the words onto the buffer
    for (int i = 0; i < length; ++i) {
      //put the token to buffer
      c.buffer.add(i);
    }

    return c;
  }

  @Override
  public boolean canApply(NEConfiguration c, String t) {
    int nBuffer = c.getBufferSize();
    return nBuffer > 0;
  }

  @Override
  public void apply(NEConfiguration c, String t) {
    c.shift(t.substring(2, t.length() - 1));
  }

  // O(n) implementation
  @Override
  public String getOracle(NEConfiguration c, Sequence dner) {
    String label = dner.get(0)[0];
    return "S(" + label + ")";
  }

}
