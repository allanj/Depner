package org.statnlp.allan.nner;

import edu.stanford.nlp.ling.CoreLabel;

/**
 * The current sentence with/without POS tag.
 * 
 * @author Allan Jie
 *
 */
public class Sent extends Sequence {

	public Sent(int capacity) {
		super(capacity);
	}
	
	public Sent(Sequence sent) {
		super(sent);
	}

	@Override
	public void set(int pos, String... tokens) {
		this.tokens[pos].setWord(tokens[0]);
		if(tokens.length==2)
			this.tokens[pos].setTag(tokens[1]);
		if(tokens.length > 2)
			throw new RuntimeException("tokens length is larger than 2");
	}

	@Override
	public String[] get(int pos) {
		CoreLabel label = this.tokens[pos];
		if(label.tag()!=null)
			return new String[]{label.word(), label.tag()};
		return new String[]{label.word()};
	}

}
