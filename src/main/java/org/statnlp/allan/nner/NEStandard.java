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

	public NEStandard(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
		super(tlp, labels, verbose);
	}

	@Override
	public boolean isTerminal(NEConfiguration c) {
		// means all words are shifted
		return c.getBufferSize() == 0;
	}

	@Override
	public void makeTransitions() {
		transitions = new ArrayList<>();
		// only have shift operation
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
			// put the token to buffer
			c.buffer.add(i);
		}

		return c;
	}

	@Override
	public boolean canApply(NEConfiguration c, String t) {
		// TODO: I should define some NER recognition rules here.
		//copy the rule from my previous code if I use IOBES encoding
		int nBuffer = c.getBufferSize();
		if (nBuffer <= 0) return false;
		
		int s0 = c.getStack(0);
		String currentLabel = t.substring(2, t.length()-1);
		if (s0 != NEConfig.NONEXIST){
			String prevLabel = c.getLabel(s0);
			if (!prevLabel.equals(NEConfig.NULL)){
				if(prevLabel.startsWith("B") && currentLabel.startsWith("I") && !prevLabel.substring(1).equals(currentLabel.substring(1)))
					return false;
				if(prevLabel.startsWith("I") && currentLabel.startsWith("I") && !prevLabel.substring(1).equals(currentLabel.substring(1)))
					return false;
				if(prevLabel.startsWith("O") && currentLabel.startsWith("I"))
					return false;
			}else{
				if(currentLabel.startsWith("I"))
					return false;
			}
		}else{
			if(currentLabel.startsWith("I"))
				return false;
		}
		
		
		
		return true;
	}

	@Override
	public void apply(NEConfiguration c, String t) {
		c.shift(t.substring(2, t.length() - 1));
	}

	// O(n) implementation
	@Override
	public String getOracle(NEConfiguration c, Sequence dner) {
		int b0 = c.getBuffer(0);
		String label = dner.get(b0)[0];
		return "S(" + label + ")";
	}

}
