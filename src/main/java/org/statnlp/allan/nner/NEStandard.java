package org.statnlp.allan.nner;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.CoreMap;

import java.util.ArrayList;
import java.util.List;


/**
 * Define an NE standard transition-based parser for named entity recognition 
 * A very basic transition-based approach for named entity recognition.
 *
 * @author Allan Jie
 */
public class NEStandard extends NERParsingSystem {
  private boolean singleRoot = true;

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

  //NOTE: unused. need to check the correctness again.
  public boolean canReach(NEConfiguration c, Sequence dner) {
	    int n = c.getSentenceSize();
	    for (int i = 1; i <= n; ++i)
	      if (c.getHead(i) != Config.NONEXIST && c.getHead(i) != dTree.getHead(i))
	        return false;

	    boolean[] inBuffer = new boolean[n + 1];
	    boolean[] depInList = new boolean[n + 1];

	    int[] leftL = new int[n + 2];
	    int[] rightL = new int[n + 2];

	    for (int i = 0; i < c.getBufferSize(); ++i)
	      inBuffer[c.buffer.get(i)] = true;

	    int nLeft = c.getStackSize();
	    for (int i = 0; i < nLeft; ++i) {
	      int x = c.stack.get(i);
	      leftL[nLeft - i] = x;
	      if (x > 0) depInList[dTree.getHead(x)] = true;
	    }

	    int nRight = 1;
	    rightL[nRight] = leftL[1];
	    for (int i = 0; i < c.getBufferSize(); ++i) {
	      boolean inList = false;
	      int x = c.buffer.get(i);
	      if (!inBuffer[dTree.getHead(x)] || depInList[x]) {
	        rightL[++nRight] = x;
	        depInList[dTree.getHead(x)] = true;
	      }
	    }

	    int[][] g = new int[nLeft + 1][nRight + 1];
	    for (int i = 1; i <= nLeft; ++i)
	      for (int j = 1; j <= nRight; ++j)
	        g[i][j] = -1;

	    g[1][1] = leftL[1];
	    for (int i = 1; i <= nLeft; ++i)
	      for (int j = 1; j <= nRight; ++j)
	        if (g[i][j] != -1) {
	          int x = g[i][j];
	          if (j < nRight && dTree.getHead(rightL[j + 1]) == x) g[i][j + 1] = x;
	          if (j < nRight && dTree.getHead(x) == rightL[j + 1]) g[i][j + 1] = rightL[j + 1];
	          if (i < nLeft && dTree.getHead(leftL[i + 1]) == x) g[i + 1][j] = x;
	          if (i < nLeft && dTree.getHead(x) == leftL[i + 1]) g[i + 1][j] = leftL[i + 1];
	        }
	    return g[nLeft][nRight] != -1;
	  }
  
  @Override
  public boolean isOracle(NEConfiguration c, String t, Sequence dner) {
    if (!canApply(c, t))
      return false;


    NEConfiguration ct = new NEConfiguration(c);
    apply(ct, t);
    return canReach(ct, dner);
  }
}
