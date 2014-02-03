package iais.network;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.TwoDimensionalMap;

public class ReconContextCostAndGradient extends AbstractCachingDiffFunction{

	RNNPhraseModel model;
	List<Tree> trainingBatch;

	public ReconContextCostAndGradient(RNNPhraseModel model, List<Tree> trainingBatch){
		this.model = model;
		this.trainingBatch = trainingBatch;
	}

	@Override
	public int domainDimension() {
		return model.totalParamSize();
	}

	@Override
	protected void calculate(double[] theta) {
		model.vectorToParams(theta);

		TwoDimensionalMap<String, String, SimpleMatrix> encodeTD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.encodeTransform) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			encodeTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		TwoDimensionalMap<String, String, SimpleMatrix> decodeTD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.decodeTransform) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			decodeTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		List<Tree> forwardPropTrees = Generics.newArrayList();


		for (int i=0; i<trainingBatch.size(); i++){	
			if(trainingBatch.get(i).isLeaf())
				continue;

			Tree tree = trainingBatch.get(i); //sentence tree
			//			List<Tree> phraseTrees = getPhraseTrees(tree, 2, 6);
			//			for(Tree pTree : phraseTrees){				

			Tree trainingTree = tree.deepCopy();
			List<Tree> leaves = trainingTree.getLeaves();

			try{
				forwardPropagateTree(trainingTree, leaves); //calculate nodevectors
			}catch(AssertionError e){
				//					System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
				continue;
			}
			//				attachRecError(trainingTree);

			//add trees and nodes for backpropagation
			forwardPropTrees.add(trainingTree);
			//			}
		}


		//backpropagate error and derivatives for each word, verb example
		//currently assuming that batch size should be one, i.e. one sentence 
		int n_nodes = 0;
		double error = 0.0;
		for (int i=0; i<forwardPropTrees.size();i++) {			
			backpropSubtrees(forwardPropTrees.get(i), forwardPropTrees.get(i).getLeaves(), encodeTD, decodeTD);	
			//			backpropDerivativesAndError(forwardPropTrees.get(i), encodeTD, decodeTD);
			//sum error accross all nodes
			double thisTreeError = sumError(forwardPropTrees.get(i));
//			n_nodes = forwardPropTrees.get(i).getLeaves().size() - 1; 
//			thisTreeError /= n_nodes;
			error += thisTreeError;
		}

		// scale the error by the number of sentences so that the
		// regularization isn't drowned out for large training batchs
		//		double scale = (1.0 / trainingBatch.size());
		Integer nTrees = 1;
		if(forwardPropTrees.size() != 0)
			nTrees = forwardPropTrees.size();
		double scale = (1.0 / nTrees);
		value = error * scale;

		value += scaleAndRegularize(encodeTD, model.encodeTransform, scale, model.op.trainOptions.regTransform);
		value += scaleAndRegularize(decodeTD, model.decodeTransform, scale, model.op.trainOptions.regTransform);

		derivative = RNNUtils.paramsToVector(theta.length, encodeTD.valueIterator(),
				decodeTD.valueIterator());

	}



	/*recursive function for unfolding a tree till its terminal nodes, attached RAE vector at each node
	 * top-down fashion of propagation */
	private void unfoldedDecode(Tree tree) {
		SimpleMatrix reconstructedVector = null;

		if (tree.isLeaf()) {			
			return;

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {

			String leftCategory = tree.children()[0].label().value();
			String rightCategory = tree.children()[1].label().value();
			SimpleMatrix Wd = model.getDecodeTransform(leftCategory, rightCategory);			

			SimpleMatrix nodeVector = RNNCoreAnnotations.getRAEVector(tree);
			reconstructedVector = RNNUtils.elementwiseApplyTanh(Wd.mult(RNNUtils.concatenateWithBias(nodeVector)));	

			if(model.op.nWordsInContext > 0){
				SimpleMatrix leftContextVecs = reconstructedVector.extractMatrix(0, model.op.nWordsInContext*model.numHid, 0, 1);
				SimpleMatrix leftVector = reconstructedVector.extractMatrix(model.op.nWordsInContext*model.numHid, (model.op.nWordsInContext+1)*model.numHid, 0, 1);
				SimpleMatrix rightVector = reconstructedVector.extractMatrix((model.op.nWordsInContext+1)*model.numHid, (model.op.nWordsInContext+2)*model.numHid, 0, 1);
				SimpleMatrix rightContextVecs = reconstructedVector.extractMatrix((model.op.nWordsInContext+2)*model.numHid, (2*model.op.nWordsInContext+2)*model.numHid, 0, 1);
				RNNCoreAnnotations.setRAEVector(tree.children()[0], leftVector);
				RNNCoreAnnotations.setRAEVector(tree.children()[1], rightVector);
				RNNCoreAnnotations.setLeftContextVecs(tree, leftContextVecs);
				RNNCoreAnnotations.setRightContextVecs(tree, rightContextVecs);

			}else{
				SimpleMatrix leftVector = reconstructedVector.extractMatrix(0, model.numHid, 0, 1);
				SimpleMatrix rightVector = reconstructedVector.extractMatrix(model.numHid, 2*model.numHid, 0, 1);
				RNNCoreAnnotations.setRAEVector(tree.children()[0], leftVector);
				RNNCoreAnnotations.setRAEVector(tree.children()[1], rightVector);
			}
			unfoldedDecode(tree.children()[0]);
			unfoldedDecode(tree.children()[1]);		

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}
	}

	public double sumError(Tree tree) {
		if (tree.isLeaf()) {
			return RNNCoreAnnotations.getReconstructionError(tree); //rec error for leaves is 0.0

		} else {
			double error = 0.0;
			for (Tree child : tree.children()) {
				error += sumError(child);
			}
			return RNNCoreAnnotations.getReconstructionError(tree) + error;
		}
	}

	public double scaleAndRegularize(TwoDimensionalMap<String, String, SimpleMatrix> derivatives,
			TwoDimensionalMap<String, String, SimpleMatrix> currentMatrices,
			double scale,
			double regCost) {
		double cost = 0.0; // the regularization cost
		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : currentMatrices) {
			SimpleMatrix D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
			D = D.scale(scale).plus(entry.getValue().scale(regCost));
			derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
			cost += entry.getValue().elementMult(entry.getValue()).elementSum() * regCost / 2.0;
		}
		return cost;
	}

	public List<Tree> getPhraseTrees(Tree tree, Integer minPhLen, Integer maxPhLen) {
		List<Tree> phraseTrees = Generics.newArrayList();
		Iterator<Tree> nodes = tree.iterator();
		while(nodes.hasNext()){
			Tree node = nodes.next();
			if(node.getLeaves().size() >= minPhLen  && node.getLeaves().size() <= maxPhLen){
				phraseTrees.add(node);
			}		
		}
		return phraseTrees;
	}

	/**
	 * This is the method to call for assigning  node vectors
	 * to the Tree.  After calling this, each of the non-leaf nodes will
	 * have the node vector.  The annotations filled in are
	 * the RNNCoreAnnotations.NodeVector. 
	 * @param leaves 
	 */
	public void forwardPropagateTree(Tree tree, List<Tree> leaves) {
		SimpleMatrix nodeVector = null;


		if (tree.isLeaf()) {			
			String word = tree.label().value();	
			if(word == null)
				throw new AssertionError("Tree containing a null word");
			nodeVector = model.getWordVector(word);	

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			forwardPropagateTree(tree.children()[0], leaves);
			forwardPropagateTree(tree.children()[1], leaves);

			String leftCategory = tree.children()[0].label().value();
			String rightCategory = tree.children()[1].label().value();
			SimpleMatrix W = model.getEncodeTransform(leftCategory, rightCategory);			

			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = null;

			if(model.op.nWordsInContext > 0){
				SimpleMatrix leftContextVecs = getContextVec(tree, leaves, true, model.op.nWordsInContext);
				SimpleMatrix rightContextVecs = getContextVec(tree, leaves, false, model.op.nWordsInContext);
				childrenVector = RNNUtils.concatenateWithBias(leftContextVecs, leftVector, rightVector, rightContextVecs);
			}else{
				childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector );
			}

			nodeVector = RNNUtils.elementwiseApplyTanh(W.mult(childrenVector));			

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}

		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);

	}

	private SimpleMatrix getContextVec(Tree tree, List<Tree> leaves, boolean isLeftContext, int nWordsInContext) {
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
			if(nodeId +1 > spannedLeaves.size()-1)
				word = RNNPhraseModel.PADDING;
			else
				word = leaves.get(nodeId+1).label().value();
			contextVecs = model.getWordVector(word);	
			for(int i=1; i<nWordsInContext; i++){				
				if(nodeId + (i+1) >spannedLeaves.size()-1)
					word = RNNPhraseModel.PADDING;
				else
					word = leaves.get(nodeId+(i+1)).label().value();
				contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
			}
		}

		return contextVecs;
	}

	private void backpropSubtrees(Tree tree, 
			List<Tree> leaves, TwoDimensionalMap<String, String, SimpleMatrix> encodeTD, 
			TwoDimensionalMap<String, String, SimpleMatrix> decodeTD){		
		if(tree.isLeaf()){
			RNNCoreAnnotations.setReconstructionError(tree, 0.0);
		}
		else{
			RNNCoreAnnotations.setRAEVector(tree, RNNCoreAnnotations.getNodeVector(tree));

			unfoldedDecode(tree); //attaches RAE vector to all nodes of this tree, starting from this node's NodeVector

			RNNCoreAnnotations.setReconstructionError(tree, getError(tree, leaves)); //getError(tree);

			bPropDecodingSide(tree, leaves, decodeTD);
			bPropEncodingSide(tree, leaves, encodeTD);
			//			backpropUnfolded(tree, encodeTD, decodeTD); //unfolded backpropagation, starting from this node

			backpropSubtrees(tree.children()[0], leaves, encodeTD, decodeTD);
			backpropSubtrees(tree.children()[1], leaves, encodeTD, decodeTD);
		}

	}


	/* returns reconstruction error for this node*/
	private double getError(Tree tree, List<Tree> rootLeaves) {
		//calculate this node's error
		double error = 0.0;
		SimpleMatrix diff =  null;
		SimpleMatrix thisDiff = null;
		List<Tree> leaves = tree.getLeaves();


		for(Tree node : leaves){

			thisDiff = RNNCoreAnnotations.getRAEVector(node).minus(RNNCoreAnnotations.getNodeVector(node));
			if(diff == null){
				diff = thisDiff;
			}else{
				diff = RNNUtils.concatenate(diff, thisDiff);
			}
		}	

		if(model.op.nWordsInContext > 0){
			SimpleMatrix leftContextVecs = getContextVec(tree, rootLeaves, true, model.op.nWordsInContext);
			SimpleMatrix rightContextVecs = getContextVec(tree, rootLeaves, false, model.op.nWordsInContext);
			SimpleMatrix leftRAEContextVecs = RNNCoreAnnotations.getLeftContextVecs(tree);
			SimpleMatrix rightRAEContextVecs = RNNCoreAnnotations.getRightContextVecs(tree);
			thisDiff = RNNUtils.concatenate(leftRAEContextVecs, rightRAEContextVecs).minus(RNNUtils.concatenate(leftContextVecs, rightContextVecs));
			diff = RNNUtils.concatenate(diff, thisDiff);			
		}

		diff = diff.elementMult(diff); 
		error = 0.5 * diff.elementSum();
		return error;		

	}

	
	private void bPropEncodingSide(Tree tree,
			List<Tree> leaves, 
			TwoDimensionalMap<String, String, SimpleMatrix> encodeTD) {

		if(tree.isLeaf()){
			return;
		}
		else{
			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());			
			SimpleMatrix encodeTransform = model.getEncodeTransform(leftCategory, rightCategory);

			SimpleMatrix deltaUp = RNNCoreAnnotations.getNodeDelta(tree);
			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			
			SimpleMatrix childrenVector = null;

			if(model.op.nWordsInContext > 0){
				SimpleMatrix leftContextVecs = getContextVec(tree, leaves, true, model.op.nWordsInContext);
				SimpleMatrix rightContextVecs = getContextVec(tree, leaves, false, model.op.nWordsInContext);
				childrenVector = RNNUtils.concatenateWithBias(leftContextVecs, leftVector, rightVector, rightContextVecs);
			}else{
				childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector );
			}			
			
			SimpleMatrix We_df = deltaUp.mult(childrenVector.transpose());
			encodeTD.put(leftCategory, rightCategory, encodeTD.get(leftCategory, rightCategory).plus(We_df));

			//calculate this node's children new delta to be used by subsequent nodes
			SimpleMatrix deltaDown = encodeTransform.transpose().mult(deltaUp);	
			SimpleMatrix leftDerivative = RNNUtils.elementwiseApplyTanhDerivative(leftVector);
			SimpleMatrix rightDerivative = RNNUtils.elementwiseApplyTanhDerivative(rightVector);
			
			SimpleMatrix leftDeltaDown = null;
			SimpleMatrix rightDeltaDown = null;
			
			leftDeltaDown = deltaDown.extractMatrix(model.op.nWordsInContext*model.numHid, (model.op.nWordsInContext+1)*model.numHid, 0, 1);
			rightDeltaDown = deltaDown.extractMatrix((model.op.nWordsInContext+1)*model.numHid, (model.op.nWordsInContext+2)*model.numHid, 0, 1);
			
			RNNCoreAnnotations.setNodeDelta(tree.children()[0], leftDerivative.elementMult(leftDeltaDown));
			RNNCoreAnnotations.setNodeDelta(tree.children()[1], rightDerivative.elementMult(rightDeltaDown));

			bPropEncodingSide(tree.children()[0], leaves, encodeTD);
			bPropEncodingSide(tree.children()[1], leaves, encodeTD);			

		}

	}

	private void bPropDecodingSide(Tree tree, List<Tree> leaves, TwoDimensionalMap<String, String, SimpleMatrix> decodeTD) {

		SimpleMatrix delta = null;
		//		double error = 0.0;

		SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);
		SimpleMatrix raeVector = RNNCoreAnnotations.getRAEVector(tree);
		SimpleMatrix currentNodeDerivative = RNNUtils.elementwiseApplyTanhDerivative(raeVector);		

		if(tree.isLeaf()){	
			delta = raeVector.minus(nodeVector);			
			delta = delta.elementMult(currentNodeDerivative);					
		}
		else{
			//recursively reach terminal nodes, bottom up fashion
			bPropDecodingSide(tree.children()[0], leaves, decodeTD);
			bPropDecodingSide(tree.children()[1], leaves, decodeTD);		

			SimpleMatrix deltaL = RNNCoreAnnotations.getNodeDelta(tree.children()[0]);
			SimpleMatrix deltaR = RNNCoreAnnotations.getNodeDelta(tree.children()[1]);
			SimpleMatrix deltaM = null;
			if(model.op.nWordsInContext>0){
				SimpleMatrix leftContextVecs = getContextVec(tree, leaves, true, model.op.nWordsInContext);
				SimpleMatrix rightContextVecs = getContextVec(tree, leaves, false, model.op.nWordsInContext);
				SimpleMatrix leftRAEContextVecs = RNNCoreAnnotations.getLeftContextVecs(tree);
				SimpleMatrix rightRAEContextVecs = RNNCoreAnnotations.getRightContextVecs(tree);
				SimpleMatrix deltaM_LContext = leftRAEContextVecs.minus(leftContextVecs);		
				SimpleMatrix deltaM_RContext = rightRAEContextVecs.minus(rightContextVecs);	
				SimpleMatrix leftNodeDerivative = RNNUtils.elementwiseApplyTanhDerivative(leftRAEContextVecs);	
				SimpleMatrix rightNodeDerivative = RNNUtils.elementwiseApplyTanhDerivative(rightRAEContextVecs);	
				
				deltaM_LContext  = deltaM_LContext.elementMult(leftNodeDerivative);	
				deltaM_RContext  = deltaM_RContext.elementMult(rightNodeDerivative);	
				
				deltaM = RNNUtils.concatenate(deltaM_LContext, deltaL, deltaR, deltaM_RContext);
			}else{
				deltaM = RNNUtils.concatenate(deltaL, deltaR);
			}

			//add gradient contribution for this node
			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());
			SimpleMatrix Wd_df = deltaM.mult(RNNUtils.concatenateWithBias(raeVector).transpose());
			decodeTD.put(leftCategory, rightCategory, decodeTD.get(leftCategory, rightCategory).plus(Wd_df));

			SimpleMatrix decodeTransform = model.getDecodeTransform(leftCategory, rightCategory);
			delta = decodeTransform.transpose().mult(deltaM);	
			delta = delta.extractMatrix(0, model.op.numHid, 0, 1).elementMult(currentNodeDerivative);


		}

		RNNCoreAnnotations.setNodeDelta(tree, delta);
		//		RNNCoreAnnotations.setReconstructionError(tree, error);

	}


}
