package org.statnlp.allan.nner;

import edu.stanford.nlp.ling.CoreLabel;

public class NESeq extends Sequence {

	public NESeq(int capacity) {
		super(capacity);
	}
	
	public NESeq(Sequence neseq) {
		super(neseq);
	}
	
	public NESeq(CoreLabel[] tokens){
		super(tokens);
	}

	@Override
	public void set(int pos, String... tokens) {
		this.tokens[pos].setNER(tokens[0]);
	}

	@Override
	public String[] get(int pos) {
		return new String[]{this.tokens[pos].ner()};
	}

}
