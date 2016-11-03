package org.statnlp.allan.nner;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;

/**
 * Pure sentence class with CoreLabel.
 * can be word sequence and named entity sequence
 * 
 * @author Allan Jie
 *
 */
public abstract class Sequence {

	protected CoreLabel[] tokens;
	protected int capacity;
	
	public Sequence(int capacity){
		this.capacity = capacity;
		this.tokens = new CoreLabel[capacity];
	}
	
	public Sequence(Sequence seq){
		this.capacity = seq.capacity;
		this.tokens = seq.tokens.clone();
	}
	
	public int size(){
		return this.tokens.length;
	}
	
	
	public abstract void set(int pos, String... tokens);
	
	public abstract String[] get(int pos);
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(this.getClass().getSimpleName()+" {");
		for(int i = 0; i< this.size(); i++){
			sb.append(" [");
			String[] out = this.get(i);
			for(int j = 0; j < out.length; j++){
				if(j==out.length-1)
					sb.append(out[j]);
				else sb.append(out[j]+",");
			}
			sb.append(" ]");
		}
		sb.append("}\n");
		return sb.toString();
	}
	
	
	
}
