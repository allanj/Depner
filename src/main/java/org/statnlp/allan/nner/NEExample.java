package org.statnlp.allan.nner;

import java.util.List;

/**
 * @author Christopher Manning
 */
public class NEExample {

  private final List<List<Integer>> features;
  private final List<String> actionSeq;

  public NEExample(List<List<Integer>> features, List<String> actionSeq) {
    this.features = features;
    this.actionSeq = actionSeq;
  }

  public List<List<Integer>> getFeatures() {
    return features;
  }

  public List<String> getActionSequences() {
    return actionSeq;
  }

}
