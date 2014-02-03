package iais.execs;

import iais.io.SRLUtils;
import iais.network.RNNPhraseModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;

/**
 * Class for performing model evaluations for comparing different models
 * 
 * @author bhanu
 *
 */

public class CompareModels {
	
	public double evaluate(RNNPhraseModel raeModel, RNNPhraseModel raecsModel, List<Tree> trees){
		double score = 0.0;
		
		SimpleMatrix wScore = raecsModel.getWScore("", "");
		SimpleMatrix wHidden = raecsModel.getWHidden("", "");
		
		Set<String> vocab = getRelevantVocab(trees, raeModel.wordVectors);
//		Set<String> vocab = raeModel.wordVectors.keySet();
		
		List<Integer> allRaeRanks = Generics.newArrayList();
		List<Integer> allRaecsRanks = Generics.newArrayList();
		for(Tree tree: trees){
			Tree raeTree = tree.deepCopy();
			Tree raecsTree = tree.deepCopy();
			
			try{
				forwardPropagateTree(raeTree, raeModel); //calculate nodevectors
				forwardPropagateTree(raecsTree, raecsModel);
				}catch(AssertionError e){
//					System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
					continue;
				}
				//				attachRecError(trainingTree);

			List<Tree> leaves = tree.getLeaves();
			predictRightNeighbourRanks(raeModel, wScore, wHidden, raeTree, leaves, allRaeRanks, vocab);
			predictRightNeighbourRanks(raecsModel, wScore, wHidden, raecsTree, leaves, allRaecsRanks, vocab);
		}

		int raeCorrect =0;
		int raecsCorrect = 0;
		int nEquals = 0;
		for(int i =0; i< allRaeRanks.size(); i++){
			if(allRaecsRanks.get(i) < allRaeRanks.get(i)){
				raecsCorrect += 1;
			}else if( allRaecsRanks.get(i) == allRaeRanks.get(i)){
				nEquals += 1;
			}else if(allRaecsRanks.get(i) > allRaeRanks.get(i)){
				raeCorrect += 1;
			}
		}
//		System.out.println("Number of times predicted rank of model is better than other model: ");
//		System.out.format("RAE CS : %d\n", raecsCorrect);		
//		System.out.format("RAE    : %d\n", raeCorrect);
//		System.out.format("Number of times Equal ranks    : %d\n", nEquals);
		double nCorrect = (raecsCorrect - raeCorrect);
//		System.out.println("Percentage Higher: "+ nCorrect/(raecsCorrect+raeCorrect));
		score = nCorrect/(raecsCorrect+raeCorrect);
		return score;
	}

	public  void runComp(String[] args){
		String raeModelPath = args[0];
		String raecsModelPath = args[1];
		String compModelPath = args[2];
		String treePath = args[3];
		List<Tree> trees = SRLUtils.readTrees(treePath).subList(0, 100);
		
		RNNPhraseModel raeModel = RNNPhraseModel.loadSerialized(raeModelPath);
		RNNPhraseModel raecsModel = RNNPhraseModel.loadSerialized(raecsModelPath);
		RNNPhraseModel compatibilityModel = RNNPhraseModel.loadSerialized(compModelPath);
//		SimpleMatrix wScore = compatibilityModel.getWScore("", "");
//		SimpleMatrix wHidden = compatibilityModel.getWHidden("", "");
		SimpleMatrix wScore = raecsModel.getWScore("", "");
		SimpleMatrix wHidden = raecsModel.getWHidden("", "");
		
//		SimpleMatrix wScore = SimpleMatrix.random(1, (raecsModel.nHidden + 1),-0.1, 0.1, raecsModel.rand);
//		SimpleMatrix wHidden = SimpleMatrix.random(raecsModel.nHidden, (2*raecsModel.op.nWordsInContext + 1)*(raecsModel.numHid)+1,-0.1, 0.1, raecsModel.rand);
		
		Set<String> vocab = getRelevantVocab(trees, raeModel.wordVectors);
//		Set<String> vocab = raeModel.wordVectors.keySet();
		
		List<Integer> allRaeRanks = Generics.newArrayList();
		List<Integer> allRaecsRanks = Generics.newArrayList();
		for(Tree tree: trees){
			Tree raeTree = tree.deepCopy();
			Tree raecsTree = tree.deepCopy();
			
			try{
				forwardPropagateTree(raeTree, raeModel); //calculate nodevectors
				forwardPropagateTree(raecsTree, raecsModel);
				}catch(AssertionError e){
//					System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
					continue;
				}
				//				attachRecError(trainingTree);

			List<Tree> leaves = tree.getLeaves();
			predictRightNeighbourRanks(raeModel, wScore, wHidden, raeTree, leaves, allRaeRanks, vocab);
			predictRightNeighbourRanks(raecsModel, wScore, wHidden, raecsTree, leaves, allRaecsRanks, vocab);
		}

		int raeCorrect =0;
		int raecsCorrect = 0;
		int nEquals = 0;
		for(int i =0; i< allRaeRanks.size(); i++){
			if(allRaecsRanks.get(i) < allRaeRanks.get(i)){
				raecsCorrect += 1;
			}else if( allRaecsRanks.get(i) == allRaeRanks.get(i)){
				nEquals += 1;
			}else if(allRaecsRanks.get(i) > allRaeRanks.get(i)){
				raeCorrect += 1;
			}
		}
		System.out.println("Number of times predicted rank of model is better than other model: ");
		System.out.format("RAE CS : %d\n", raecsCorrect);		
		System.out.format("RAE    : %d\n", raeCorrect);
		System.out.format("Number of times Equal ranks    : %d\n", nEquals);
		double nCorrect = (raecsCorrect - raeCorrect);
		System.out.println("Percentage Higher: "+ nCorrect/(raecsCorrect+raeCorrect));
	}

	private  Set<String> getRelevantVocab(List<Tree> trees, Map<String, SimpleMatrix> wordVectors) {
		Set<String> vocab = Generics.newHashSet();
		for(Tree tree : trees){
			List<Tree> leaves = tree.getLeaves();
			for(Tree leaf : leaves){
				CoreLabel label = (CoreLabel)leaf.label();
				String word = label.value();
				vocab.add(word);
			}
		}
		vocab.add(RNNPhraseModel.PADDING);
		vocab.add(RNNPhraseModel.UNKNOWN_WORD);
		
		return vocab;
	}

	private  void predictRightNeighbourRanks(RNNPhraseModel model, SimpleMatrix wScore, 
			SimpleMatrix wHidden, Tree tree, List<Tree> leaves, 
			List<Integer> ranks, Set<String> vocab) {
		if(tree.isLeaf()){
			return;
		}else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			
			SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);			
			
			//for this phrase and its left context calculate probability for each word in vocab		
			SimpleMatrix scores = new SimpleMatrix(vocab.size(), 1);
			Map<Integer, String> wordIdMap = Generics.newHashMap();
			Map<String, Double> scoresMap = new TreeMap<String, Double>();
			int i = 0;
			SimpleMatrix leftContextVecs = getContextVec(model, tree, leaves, true, model.op.nWordsInContext);
			SimpleMatrix rightContextVecs = getContextVec(model, tree, leaves, false, model.op.nWordsInContext);
			rightContextVecs = rightContextVecs.extractMatrix(0, (model.op.nWordsInContext-1)*model.numHid, 0, 1);
			for(String word : vocab){		
				SimpleMatrix wordRightContextVecs = null;
				if(rightContextVecs==null){
					wordRightContextVecs = model.getWordVector(word);
				}
				else{
					wordRightContextVecs = RNNUtils.concatenate(rightContextVecs, model.getWordVector(word));
				}
				SimpleMatrix hiddenValues = wHidden.mult(
						RNNUtils.concatenateWithBias(leftContextVecs, nodeVector, wordRightContextVecs));
				hiddenValues = RNNUtils.elementwiseApplyTanh(hiddenValues);
				SimpleMatrix score = wScore.mult(RNNUtils.concatenateWithBias(hiddenValues));
//				score = RNNUtils.softmax(score);
				scores.set(i,0, score.get(0, 0));
				wordIdMap.put(i, word);			
//				scoresMap.put(word, score.get(0, 0));
				i++;				
			}
			
			scores = RNNUtils.softmax(scores);
			for(int id =0; id < scores.numRows(); id++){				
				String word = wordIdMap.get(id);
				scoresMap.put(word, scores.get(id, 0));
			}
			
			double total = scores.elementSum();
			// Get entries and sort them.
			List<Entry<String, Double>> entries = new ArrayList<Entry<String, Double>>(scoresMap.entrySet());
			Collections.sort(entries, new Comparator<Entry<String, Double>>() {
			    public int compare(Entry<String, Double> e1, Entry<String, Double> e2) {
			        return e2.getValue().compareTo(e1.getValue());
			    }
			});
			//find rank of correct right most  neighbour
			int r = 0;
			String rightNeighbour = getRightmostNeighbour(tree, leaves, model.op.nWordsInContext);
//			System.out.println(model.toString() +" "+ "Actual: "+rightNeighbour + " Predicted: "+ entries.get(0).getKey());
			
			for(Entry<String, Double> entry : entries){
				if(entry.getKey().equalsIgnoreCase(rightNeighbour)){
					break;
				}
				r++;
			}
			ranks.add(r);
			
			
			predictRightNeighbourRanks(model, wScore, wHidden, tree.children()[0], leaves, ranks, vocab);
			predictRightNeighbourRanks(model, wScore, wHidden, tree.children()[1], leaves, ranks, vocab);
			
		}
		
	}
	
	private  String getRightmostNeighbour(Tree tree, List<Tree> leaves, int nWordsInContext) {
		
		List<Tree> spannedLeaves = tree.getLeaves();
		Tree leafToUse = spannedLeaves.get(spannedLeaves.size()-1);
		CoreLabel label = (CoreLabel)leafToUse.label();
		Integer nodeId = label.get(IndexAnnotation.class) - 1; // node indexes starts from 1
		String word = "";
		if(nodeId + nWordsInContext  > leaves.size()-1)
			word = RNNPhraseModel.PADDING;
		else
			word = leaves.get(nodeId+nWordsInContext).label().value();
		
		return word;
	}

	private  SimpleMatrix getContextVec(RNNPhraseModel model, Tree tree, List<Tree> leaves, boolean isLeftContext, int nWordsInContext) {
		SimpleMatrix contextVecs = null;
		List<Tree> spannedLeaves = tree.getLeaves();
		Tree leafToUse = null;
		if(isLeftContext){
			leafToUse = spannedLeaves.get(0);
			CoreLabel label = (CoreLabel)leafToUse.label();
			Integer nodeId = label.get(IndexAnnotation.class) - 1; // node indexes starts from 1
			String word = "";
			if(nodeId - 1 < 0)
				word = RNNPhraseModel.PADDING;
			else
				word = leaves.get(nodeId-1).label().value();
			contextVecs = model.getWordVector(word);	
			for(int i=1; i<nWordsInContext; i++){				
				if(nodeId - (i+1) < 0)
					word = RNNPhraseModel.PADDING;
				else
					word = leaves.get(nodeId-(i+1)).label().value();
				contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
			}
		}else{
			leafToUse = spannedLeaves.get(spannedLeaves.size()-1);
			CoreLabel label = (CoreLabel)leafToUse.label();
			Integer nodeId = label.get(IndexAnnotation.class) - 1; // node indexes starts from 1
			String word = "";
			if(nodeId +1 > leaves.size()-1)
				word = RNNPhraseModel.PADDING;
			else
				word = leaves.get(nodeId+1).label().value();
			contextVecs = model.getWordVector(word);	
			for(int i=1; i<nWordsInContext; i++){				
				if(nodeId + (i+1) >leaves.size()-1)
					word = RNNPhraseModel.PADDING;
				else
					word = leaves.get(nodeId+(i+1)).label().value();
				contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
			}
		}

//		contextVecs = RNNUtils.elementwiseApplyTanh(contextVecs);
		return contextVecs;
	}
	
	public  void forwardPropagateTree(Tree tree, RNNPhraseModel model) {
		SimpleMatrix nodeVector = null;


		if (tree.isLeaf()) {			
			String word = tree.label().value();	
			if(word == null)
				throw new AssertionError("Tree containing a null word");
			nodeVector = model.getWordVector(word);	

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			forwardPropagateTree(tree.children()[0], model);
			forwardPropagateTree(tree.children()[1], model);

			String leftCategory = tree.children()[0].label().value();
			String rightCategory = tree.children()[1].label().value();
			SimpleMatrix W = model.getEncodeTransform(leftCategory, rightCategory);			

			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector);

			nodeVector = RNNUtils.elementwiseApplyTanh(W.mult(childrenVector));			

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}

		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);

	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		CompareModels cm = new CompareModels();
		cm.runComp(args);

	}

}
