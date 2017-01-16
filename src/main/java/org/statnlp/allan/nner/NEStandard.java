package org.statnlp.allan.nner;

import java.util.ArrayList;
import java.util.List;


import edu.stanford.nlp.trees.TreebankLanguagePack;

/**
 * Define an NE standard transition-based parser for named entity recognition A
 * very basic transition-based approach for named entity recognition.
 *
 * @author Allan Jie
 */
public class NEStandard extends NERParsingSystem {

	private boolean singleRoot = true;
	boolean iobes;
	
	public NEStandard(TreebankLanguagePack tlp, List<String> labels, boolean verbose, boolean IOBESeconding) {
		super(tlp, labels, verbose);
		this.iobes = IOBESeconding;
	}

	@Override
	public boolean isTerminal(NEConfiguration c) {
		// means all words are shifted
		return c.getBufferSize() == 0 && c.getStackSize() == 1;
	}

	@Override
	public void makeTransitions() {
		transitions = new ArrayList<>();
		// only have shift operation
		for (String label : labels)
			transitions.add("S(" + label + ")");
		transitions.add("L");
	    transitions.add("R");
	}

	@Override
	public NEConfiguration initialConfiguration(Sequence s) {
		NEConfiguration c = new NEConfiguration(s);
		int length = s.size();
		// For each token, add dummy elements to the configuration's tree
		// and add the words onto the buffer
		for (int i = 1; i < length; ++i) {
			// put the token to buffer
			c.tree.add(NEConfig.NONEXIST, NEConfig.UNKNOWN);
			c.buffer.add(i);
		}
		c.stack.add(0);
		c.ners.set(0, "O");
		return c;
	}

	@Override
	public boolean canApply(NEConfiguration c, String t) {
		
		if (t.startsWith("L") || t.startsWith("R")) {
		      int h = t.startsWith("L") ? c.getStack(0) : c.getStack(1);
		      if (h < 0) return false;
		      //if (h > 0 && label.equals(rootLabel)) return false;
		}

	    int nStack = c.getStackSize();
	    int nBuffer = c.getBufferSize();

	    if (t.startsWith("L"))
	      return nStack > 2;
	    else if (t.startsWith("R")) {
	      if (singleRoot)
	        return (nStack > 2) || (nStack == 2 && nBuffer == 0);
	      else
	        return nStack >= 2;
	    } else {
			if (nBuffer <= 0) return false;
			int bMinus1 = c.getBuffer(0) -1;
			String currentLabel = t.substring(2, t.length()-1);
			if (bMinus1 != NEConfig.NONEXIST){
				String prevLabel = c.getLabel(bMinus1);
				if (!prevLabel.equals(NEConfig.NULL)){
					if(prevLabel.startsWith("I")){
						if(currentLabel.startsWith("I") && !prevLabel.substring(1).equals(currentLabel.substring(1))) return false;
						if(currentLabel.startsWith("E") && !prevLabel.substring(1).equals(currentLabel.substring(1))) return false;
						if(iobes && (currentLabel.startsWith("O") || currentLabel.startsWith("B") || currentLabel.startsWith("S")  ) ) return false;
						
					}else if(prevLabel.equals("O")){
						if(currentLabel.startsWith("I") || currentLabel.startsWith("E")) return false;
						
					}else if(prevLabel.startsWith("B")){
						if(currentLabel.startsWith("I")  && !prevLabel.substring(1).equals(currentLabel.substring(1)) ) return false;
						if(currentLabel.startsWith("E")  && !prevLabel.substring(1).equals(currentLabel.substring(1)) ) return false;
						if(iobes && ( currentLabel.equals("O") || currentLabel.equals("B") || currentLabel.equals("S") )  ) return false;
						
					}else if(prevLabel.startsWith("E")){
						if(currentLabel.startsWith("I") || currentLabel.startsWith("E")) return false;
						
					}else if(prevLabel.startsWith("S")){
						if(currentLabel.startsWith("I") || currentLabel.startsWith("E")) return false;
						
					}else{
						throw new RuntimeException("Unknown type "+prevLabel+" in network compilation");
					}
				}else{
					if(currentLabel.startsWith("I") ||  currentLabel.startsWith("E"))
						return false;
				}
			}else{
				if(currentLabel.startsWith("I") || currentLabel.startsWith("E"))
					return false;
			}
	    }
		return true;
	}

	@Override
	public void apply(NEConfiguration c, String t) {
		int w1 = c.getStack(1);
	    int w2 = c.getStack(0);
	    if (t.startsWith("L")) {
	      c.addArc(w2, w1, NEConfig.UNKNOWN);
	      c.removeSecondTopStack();
	    } else if (t.startsWith("R")) {
	      c.addArc(w1, w2, NEConfig.UNKNOWN);
	      c.removeTopStack();
	    } else {
	    	String str = t.substring(2, t.length() - 1);
	    	if (!str.equals("O") && !str.contains("-"))
	    		throw new RuntimeException("label "+ t + "???");
	    	c.shift(str);
	    }
	}

	// O(n) implementation
	@Override
	public String getOracle(NEConfiguration c, NEDependencyTree dTree, Sequence dner) {
		int w1 = c.getStack(1);
	    int w2 = c.getStack(0);
	    if (w1 > 0 && dTree.getHead(w1) == w2)
	      return "L";
	    else if (w1 >= 0 && dTree.getHead(w2) == w1 && !c.hasOtherChild(w2, dTree))
	      return "R";
	    else {
	    	int b0 = c.getBuffer(0);
			String ner = dner.tokens[b0].ner();
			return "S(" + ner + ")";
	    }
	}
	
}
