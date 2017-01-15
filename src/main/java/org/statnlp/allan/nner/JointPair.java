package org.statnlp.allan.nner;

public class JointPair {

	public Sequence ners;
	public NEDependencyTree tree;
	
	public JointPair (Sequence ners, NEDependencyTree tree) {
		this.ners = ners;
		this.tree = tree;
	}
}
