package org.statnlp.allan.nner;

import java.util.ArrayList;
import java.util.List;

/**
 * Describe the current configuration of a nNER.
 *
 * There is no root node. and this one is not nested. Currently, we try one with
 * shift(t), shift one entity only
 *
 * @author Allan Jie
 */
public class NEConfiguration {

	final List<Integer> stack;
	final List<Integer> buffer;

	final Sequence ners;
	final Sequence sent;

	public NEConfiguration(NEConfiguration config) {
		stack = new ArrayList<>(config.stack);
		buffer = new ArrayList<>(config.buffer);
		ners = new NESeq(config.ners);
		sent = new Sent(config.sent);
	}

	public NEConfiguration(Sequence sentence) {
		this.stack = new ArrayList<>();
		this.buffer = new ArrayList<>();
		this.ners = new NESeq(sentence.size());
		this.sent = sentence;
	}

	public boolean shift(String ner) {
		int k = getBuffer(0);
		if (k == NEConfig.NONEXIST)
			return false;
		buffer.remove(0);
		stack.add(k);
		ners.set(k, ner);
		return true;
	}

	/**
	 * Probably used in left-arc operation
	 * 
	 * @return
	 */
	public boolean removeSecondTopStack() {
		int nStack = getStackSize();
		if (nStack < 2)
			return false;
		stack.remove(nStack - 2);
		return true;
	}

	public boolean removeTopStack() {
		int nStack = getStackSize();
		if (nStack < 1)
			return false;
		stack.remove(nStack - 1);
		return true;
	}

	public int getStackSize() {
		return stack.size();
	}

	public int getBufferSize() {
		return buffer.size();
	}

	public int getSentenceSize() {
		return sent.size();
	}

	/**
	 * @param k
	 *            Word index (indexed from 0)
	 */
	public String getLabel(int k) {
		return k < 0 || k >= sent.size() ? NEConfig.NULL : ners.get(k)[0];
	}

	/**
	 * Get the sentence index of the kth word on the stack.
	 *
	 * @return Sentence index or {@link NEConfig#NONEXIST} if stack doesn't have
	 *         an element at this index
	 */
	public int getStack(int k) {
		int nStack = getStackSize();
		return (k >= 0 && k < nStack) ? stack.get(nStack - 1 - k) : NEConfig.NONEXIST;
	}

	/**
	 * Get the sentence index of the kth word on the buffer.
	 *
	 * @return Sentence index or {@link NEConfig#NONEXIST} if stack doesn't have
	 *         an element at this index
	 */
	public int getBuffer(int k) {
		return (k >= 0 && k < getBufferSize()) ? buffer.get(k) : NEConfig.NONEXIST;
	}

	/**
	 * @param k
	 *            Word index (indexed from 0)
	 */
	public String getWord(int k) {
		return k < 0 || k >= sent.size() ? NEConfig.NULL : sent.get(k)[0];
	}

	/**
	 * @param k
	 *            Word index (0 indexed)
	 */
	public String getPOS(int k) {
		return k < 0 || k >= sent.size() ? NEConfig.NULL : sent.get(k)[1];
	}

	/**
	 * Returns a string that concatenates all elements on the stack and buffer,
	 * as well as named entities
	 * 
	 * @return
	 */
	public String getStr() {
		String s = "[S] ";
		for (int i = 0; i < getStackSize(); ++i) {
			if (i > 0)
				s = s + ",";
			s = s + stack.get(i); // this is the word index
		}
		s = s + "[B] ";
		for (int i = 0; i < getBufferSize(); ++i) {
			if (i > 0)
				s = s + ",";
			s = s + buffer.get(i); // this is the word index
		}
		s = s + "[NE] ";
		for (int i = 0; i < getStackSize(); ++i) {
			if (i > 0)
				s = s + ",";
			s = s + getLabel(stack.get(i));
		}
		return s;
	}
	
	@Override
	public String toString(){
		return getStr();
	}
}