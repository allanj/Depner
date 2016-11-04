package org.statnlp.allan.depner;

import java.util.List;

/**
 * @author Christopher Manning
 */
public class Example {

  private final List<Integer> feature;
  private final List<Integer> label;

  public Example(List<Integer> feature, List<Integer> label) {
    this.feature = feature;
    this.label = label;
  }

  public List<Integer> getFeature() {
    return feature;
  }

  public List<Integer> getLabel() {
    return label;
  }

}
