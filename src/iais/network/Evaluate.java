package iais.network;

import iais.io.Config;
import iais.io.SRLUtils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.POSTags;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.Categories;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.Labels;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.VerbIds;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.StringUtils;

public class Evaluate {
	final SRLCostAndGradient cag;
	final RNTNModel model;
	final SRLCostAndGradient2 cag2;
	final ReconstructionCostAndGradient cag3;
	final RNNPhraseModel rnnPhModel;
	ReconContextCostAndGradient cag4 ;
	ReconRankingCostAndGradient cag5;
	int labelsCorrect;
	int labelsIncorrect;

	// the matrix will be [gold][predicted]
	int[][] labelConfusion;

	int rootLabelsCorrect;
	int rootLabelsIncorrect;

	int[][] rootLabelConfusion;

	IntCounter<Integer> lengthLabelsCorrect;
	IntCounter<Integer> lengthLabelsIncorrect;

	private static final NumberFormat NF = new DecimalFormat("0.000000");

	public Evaluate(RNTNModel model) {
		this.model = model;
		this.cag = new SRLCostAndGradient(model, null);
		this.cag2 = null;
		this.cag3 = null;
		this.rnnPhModel = null;
		reset();
	}

	public Evaluate(RNTNModel model, Integer type){
		this.model = model;		
		this.cag2 = new SRLCostAndGradient2(model, null);		
		this.cag = null;
		this.cag3 = null;
		this.rnnPhModel = null;
		reset();
	}
	public Evaluate(RNNPhraseModel model){
		this.model = null;
		this.rnnPhModel = model;
		this.cag3 = new ReconstructionCostAndGradient(model, null);
		this.cag = null;
		this.cag2 = null;
		//		reset();
	}
	public Evaluate(RNNPhraseModel model, List<Tree> trees, Integer type){
		this.model = null;
		this.rnnPhModel = model;
		if(type==0)
			this.cag4 = new ReconContextCostAndGradient(model, null);
		else
			this.cag5 = new ReconRankingCostAndGradient(model, trees);
		this.cag = null;
		this.cag3 = null;
		this.cag2 = null;
		//		reset();
	}
	public void reset() {
		labelsCorrect = 0;
		labelsIncorrect = 0;
		labelConfusion = new int[model.op.numClasses][model.op.numClasses];

		rootLabelsCorrect = 0;
		rootLabelsIncorrect = 0;
		rootLabelConfusion = new int[model.op.numClasses][model.op.numClasses];

		lengthLabelsCorrect = new IntCounter<Integer>();
		lengthLabelsIncorrect = new IntCounter<Integer>();
	}

	public double eval(List<Tree> trees) {

		List<VerbIds> verbIndices = Generics.newArrayList();
		List<Labels> sentenceLabels = Generics.newArrayList();
		List<POSTags> sentPostags = Generics.newArrayList();
		if(trees != null){
			for (Tree tree : trees){
				CoreLabel label  = (CoreLabel)tree.label();
				verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));			
				sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
				sentPostags.add(RNNCoreAnnotations.getPOSTags(tree));
			}
		}

		List<Integer> trueLabels = Generics.newArrayList();
		List<Integer>	predictedLabels = Generics.newArrayList();
		for (int i=0; i<trees.size(); i++){	
			if((verbIndices.get(i).size() == 0) || trees.get(i).isLeaf())
				continue;
			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
				Tree tree = trees.get(i); //sentence tree
				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
				List<Tree> leaves = tree.getLeaves();
				
				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
					continue;  //sentence tokens and labels mismatch.. have to fix this

				this.cag.setPosFeatures(leaves, sentPostags.get(i));
				for(int wid=0; wid<leaves.size(); wid++){	
					Tree trainingTree = tree.deepCopy();		
					Integer trueLabel = sentenceLabels.get(i).get(nverbs).get(wid);
					this.cag.setVerbNWordIndexFeatures(trainingTree, verbIndex, wid);
					this.cag.forwardPropagateTree(trainingTree); //calculate nodevectors
					trueLabels.add(trueLabel);
					//find the subtree spanned by word and verb pair
					Tree subtree = this.cag.findSubTree(trainingTree, verbIndex, wid, trueLabel);
					//					Tree word = trainingTree.getLeaves().get(wid);
					//					Tree verb = trainingTree.getLeaves().get(verbIndex);
					List<Integer> nodeIds = this.cag.getCatInputNodes(trainingTree, subtree, model.op.windowSize, verbIndex, wid);
					// this will attach the error vectors and prediction class to the top node of the subtree
					// to each node in the tree
					this.cag.calcPredictions(trainingTree, subtree, nodeIds); //calculate prediction for this word and verb pair	

					predictedLabels.add(RNNCoreAnnotations.getPredictedClass(subtree));
				}
			}

		}

		int ncorrect = 0;
		for(int i =0; i<predictedLabels.size(); i++){
			if(predictedLabels.get(i) == trueLabels.get(i))
				ncorrect += 1;
		}
		double accuracy = 100.0*(1.0*ncorrect/trueLabels.size());
		System.out.println("Accuracy : "+accuracy);
		return accuracy;
	}

	public double eval_reconRanking(){
		double cost  = 0.0;
		double[] x = this.cag5.model.paramsToVector();
		this.cag5.model.randContext = new Random(this.cag5.model.SEED); //everytime use the same random sequence, 
		cost = this.cag5.valueAt(x);

		return cost;
	}

	public double eval_compatibility(List<Tree> trees, String trainedRAEModelName){
		double cost  = 0.0;
		RNNPhraseModel trainedModel = RNNPhraseModel.loadSerialized(trainedRAEModelName);
		SimpleMatrix We = trainedModel.getEncodeTransform("", "");
		CompatibilityCostAndGradient gcFunc = new CompatibilityCostAndGradient(rnnPhModel, We, trees);

		double[] x = gcFunc.model.paramsToVector();
		gcFunc.model.randContext = new Random(gcFunc.model.SEED); //everytime use the same random sequence, 
		cost = gcFunc.valueAt(x);

		return cost;
	}

	public double eval2(List<Tree> trees){
		Categories iobCategories = Categories.getIOBCategories();
		Categories roleCats = Categories.getRoleCats(iobCategories);

		List<VerbIds> verbIndices = Generics.newArrayList();
		List<Labels> sentenceLabels = Generics.newArrayList();

		List<Integer> allTrueLabels = Generics.newArrayList();
		List<Integer>	allPredictedLabels = Generics.newArrayList();

		if(trees != null){
			for (Tree tree : trees){
				CoreLabel label  = (CoreLabel)tree.label();
				verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));			
				sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
			}
		}

		for (int i=0; i<trees.size(); i++){	
			if((verbIndices.get(i).size() == 0) || trees.get(i).isLeaf())
				continue;
			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
				Tree tree = trees.get(i); //sentence tree
				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
				List<Tree> leaves = tree.getLeaves();
				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
					continue;  //sentence tokens and labels mismatch.. have to fix this

				Tree trainingTree = tree.deepCopy();
				this.cag2.setVerbNWordIndexFeatures(trainingTree, verbIndex);
				//				this.cag2.setAdditionalFeatures(trainingTree, verbIndex);
				List<Integer> trueLabels = sentenceLabels.get(i).get(nverbs);
				Tree verbNode = trainingTree.getLeaves().get(verbIndex);
				this.cag2.attachLabelsToNodes(trainingTree, trueLabels, iobCategories, roleCats);
				this.cag2.forwardPropagateTree(trainingTree, trainingTree, verbNode); //calculate nodevectors
				this.cag2.attachPredictions(trainingTree, trainingTree, verbNode);


				List<Integer> predictedLabels = Generics.newArrayList();
				List<Integer> newTrueLabels = Generics.newArrayList();
				for(Tree ptree : trainingTree.getLeaves()){
					predictedLabels.add(RNNCoreAnnotations.getPredictedClass(ptree));
					newTrueLabels.add(RNNCoreAnnotations.getGoldClass(ptree));
				}
				for(Integer label : predictedLabels)
					allPredictedLabels.add(label);
				for (Integer label : newTrueLabels)
					allTrueLabels.add(label);


				countTree(trainingTree);
			}

		}
		int ncorrect = 0;
		for(int i =0; i<allPredictedLabels.size(); i++){
			if(allPredictedLabels.get(i) == allTrueLabels.get(i))
				ncorrect += 1;
		}
		double accuracy = 100.0*(1.0*ncorrect/allTrueLabels.size());
		System.out.println("Accuracy : "+accuracy);
		//accuracy of internal nodes classification		
		//		System.out.println("InternalNodes accuracy: "+exactNodeAccuracy());
		//		printSummary();
		//


		return accuracy;

	}
	public void eval(Tree tree) {
		cag.forwardPropagateTree(tree);

		countTree(tree);
		countRoot(tree);
		countLengthAccuracy(tree);
	}

	private int countLengthAccuracy(Tree tree) {
		if (tree.isLeaf()) {
			return 0;
		}
		Integer gold = RNNCoreAnnotations.getGoldClass(tree);
		Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
		int length;
		if (tree.isPreTerminal()) {
			length = 1;
		} else {
			length = 0;
			for (Tree child : tree.children()) {
				length += countLengthAccuracy(child);
			}
		}
		if (gold.equals(predicted)) {
			lengthLabelsCorrect.incrementCount(length);
		} else {
			lengthLabelsIncorrect.incrementCount(length);
		}
		return length;
	}

	private void countTree(Tree tree) {
		if (tree.isLeaf()) {
			return;
		}
		for (Tree child : tree.children()) {
			countTree(child);
		}
		Integer gold = RNNCoreAnnotations.getGoldClass(tree);
		Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
		if (gold.equals(predicted)) {
			labelsCorrect++;
		} else {
			labelsIncorrect++;
		}
		labelConfusion[gold][predicted]++;
	}

	private void countRoot(Tree tree) {
		Integer gold = RNNCoreAnnotations.getGoldClass(tree);
		Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
		if (gold.equals(predicted)) {
			rootLabelsCorrect++;
		} else {
			rootLabelsIncorrect++;
		}
		rootLabelConfusion[gold][predicted]++;
	}

	public double exactNodeAccuracy() {
		return (double) labelsCorrect / ((double) (labelsCorrect + labelsIncorrect));
	}

	public double exactRootAccuracy() {
		return (double) rootLabelsCorrect / ((double) (rootLabelsCorrect + rootLabelsIncorrect));
	}

	public Counter<Integer> lengthAccuracies() {
		Set<Integer> keys = Generics.newHashSet();
		keys.addAll(lengthLabelsCorrect.keySet());
		keys.addAll(lengthLabelsIncorrect.keySet());

		Counter<Integer> results = new ClassicCounter<Integer>();
		for (Integer key : keys) {
			results.setCount(key, lengthLabelsCorrect.getCount(key) / (lengthLabelsCorrect.getCount(key) + lengthLabelsIncorrect.getCount(key)));
		}
		return results;
	}

	public void printLengthAccuracies() {
		Counter<Integer> accuracies = lengthAccuracies();
		Set<Integer> keys = Generics.newTreeSet();
		keys.addAll(accuracies.keySet());
		System.err.println("Label accuracy at various lengths:");
		for (Integer key : keys) {
			System.err.println(StringUtils.padLeft(Integer.toString(key), 4) + ": " + NF.format(accuracies.getCount(key)));
		}
	}

	private static final int[] NEG_CLASSES = {0, 1};
	private static final int[] POS_CLASSES = {3, 4};

	public double[] approxNegPosAccuracy() {
		return approxAccuracy(labelConfusion, NEG_CLASSES, POS_CLASSES);
	}

	public double approxNegPosCombinedAccuracy() {
		return approxCombinedAccuracy(labelConfusion, NEG_CLASSES, POS_CLASSES);
	}

	public double[] approxRootNegPosAccuracy() {
		return approxAccuracy(rootLabelConfusion, NEG_CLASSES, POS_CLASSES);
	}

	public double approxRootNegPosCombinedAccuracy() {
		return approxCombinedAccuracy(rootLabelConfusion, NEG_CLASSES, POS_CLASSES);
	}

	private static void printConfusionMatrix(String name, int[][] confusion) {
		System.err.println(name + " confusion matrix: rows are gold label, columns predicted label");
		for (int i = 0; i < confusion.length; ++i) {
			for (int j = 0; j < confusion[i].length; ++j) {
				System.err.print(StringUtils.padLeft(confusion[i][j], 10));
			}
			System.err.println();
		}
	}

	private static double[] approxAccuracy(int[][] confusion, int[] ... classes) {
		int[] correct = new int[classes.length];
		int[] incorrect = new int[classes.length];
		double[] results = new double[classes.length];
		for (int i = 0; i < classes.length; ++i) {
			for (int j = 0; j < classes[i].length; ++j) {
				for (int k = 0; k < classes[i].length; ++k) {
					correct[i] += confusion[classes[i][j]][classes[i][k]];
				}
			}
			for (int other = 0; other < classes.length; ++other) {
				if (other == i) {
					continue;
				}
				for (int j = 0; j < classes[i].length; ++j) {
					for (int k = 0; k < classes[other].length; ++k) {
						incorrect[i] += confusion[classes[i][j]][classes[other][k]];
						incorrect[i] += confusion[classes[other][j]][classes[i][k]];
					}
				}
			}
			results[i] = ((double) correct[i]) / ((double) (correct[i] + incorrect[i]));
		}
		return results;
	}

	private static double approxCombinedAccuracy(int[][] confusion, int[] ... classes) {
		int correct = 0;
		int incorrect = 0;
		for (int i = 0; i < classes.length; ++i) {
			for (int j = 0; j < classes[i].length; ++j) {
				for (int k = 0; k < classes[i].length; ++k) {
					correct += confusion[classes[i][j]][classes[i][k]];
				}
			}
			for (int other = 0; other < classes.length; ++other) {
				if (other == i) {
					continue;
				}
				for (int j = 0; j < classes[i].length; ++j) {
					for (int k = 0; k < classes[other].length; ++k) {
						incorrect += confusion[classes[i][j]][classes[other][k]];
						incorrect += confusion[classes[other][j]][classes[i][k]];
					}
				}
			}
		}
		return ((double) correct) / ((double) (correct + incorrect));
	}

	public void printSummary() {
		System.err.println("EVALUATION SUMMARY");
		System.err.println("Tested " + (labelsCorrect + labelsIncorrect) + " labels");
		System.err.println("  " + labelsCorrect + " correct");
		System.err.println("  " + labelsIncorrect + " incorrect");
		System.err.println("  " + NF.format(exactNodeAccuracy()) + " accuracy");
		//		System.err.println("Tested " + (rootLabelsCorrect + rootLabelsIncorrect) + " roots");
		//		System.err.println("  " + rootLabelsCorrect + " correct");
		//		System.err.println("  " + rootLabelsIncorrect + " incorrect");
		//		System.err.println("  " + NF.format(exactRootAccuracy()) + " accuracy");

		printConfusionMatrix("Label", labelConfusion);
		//		printConfusionMatrix("Root label", rootLabelConfusion);

		double[] approxLabelAccuracy = approxNegPosAccuracy();
		System.err.println("Approximate negative label accuracy: " + NF.format(approxLabelAccuracy[0]));
		System.err.println("Approximate positive label accuracy: " + NF.format(approxLabelAccuracy[1]));
		//		System.err.println("Combined approximate label accuracy: " + NF.format(approxNegPosCombinedAccuracy()));

		//		double[] approxRootLabelAccuracy = approxRootNegPosAccuracy();
		//		System.err.println("Approximate negative root label accuracy: " + NF.format(approxRootLabelAccuracy[0]));
		//		System.err.println("Approximate positive root label accuracy: " + NF.format(approxRootLabelAccuracy[1]));
		//		System.err.println("Combined approximate root label accuracy: " + NF.format(approxRootNegPosCombinedAccuracy()));

		//printLengthAccuracies();
	}

	public void create_preds_file(List<Tree> trees){
		List<VerbIds> verbIndices = Generics.newArrayList();
		List<Labels> sentenceLabels = Generics.newArrayList();

		if(trees != null){
			for (Tree tree : trees){
				CoreLabel label  = (CoreLabel)tree.label();
				verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));			
				sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
			}
		}

		List<Integer> trueLabels = Generics.newArrayList();
		List<Integer>	predictedLabels = Generics.newArrayList();
		for (int i=0; i<trees.size(); i++){	
			if((verbIndices.get(i).size() == 0) || trees.get(i).isLeaf())
				continue;
			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
				Tree tree = trees.get(i); //sentence tree
				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
				List<Tree> leaves = tree.getLeaves();
				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
					continue;  //sentence tokens and labels mismatch.. have to fix this
				//				this.cag.forwardPropagateTree(trainingTree); //calculate nodevectors
				for(int wid=0; wid<leaves.size(); wid++){	
					Tree trainingTree = tree.deepCopy();
					Integer trueLabel = sentenceLabels.get(i).get(nverbs).get(wid);
					this.cag.setVerbNWordIndexFeatures(trainingTree, verbIndex, wid);
					this.cag.forwardPropagateTree(trainingTree); //calculate nodevectors
					trueLabels.add(trueLabel);
					//find the subtree spanned by word and verb pair
					Tree subtree = this.cag.findSubTree(trainingTree, verbIndex, wid, trueLabel);
					//					Tree word = trainingTree.getLeaves().get(wid);
					//					Tree verb = trainingTree.getLeaves().get(verbIndex);
					List<Integer> nodeIds = this.cag.getCatInputNodes(trainingTree, subtree,  model.op.windowSize, verbIndex, wid);

					// this will attach the error vectors and prediction class to the top node of the subtree
					// to each node in the tree
					this.cag.calcPredictions(trainingTree, subtree, nodeIds); //calculate prediction for this word and verb pair	
					predictedLabels.add(RNNCoreAnnotations.getPredictedClass(subtree));
				}
			}

		}
		Integer nMostFreqCats = Config.NMOST_FREQUENT_CATS;  //number of classes would be 34 + 1 
		HashMap<Integer, String> categories = new HashMap<>();
		Iterable<String> catsStr = IOUtils.readLines(Config.FREQ_CATEGORIES, "utf-8");
		Integer cid = 0;
		for(String key : catsStr){
			key = key.trim();
			categories.put(cid, key);
			cid++;
			if(cid == nMostFreqCats)
				break;
		}
		categories.put(cid, Config.OTHER_KEY);
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(Config.PROJECT_HOME+"data/results/preds_srl_rntn1.txt", "UTF-8");
			Integer i = 0;
			for(Integer predLabel : predictedLabels){ 
				String catlabel = categories.get(predLabel);
				writer.println(i.toString()+"\t"+catlabel);
				i++;
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

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
	public void create_vectors_file(List<Tree> trees){
		List<VerbIds> verbIndices = Generics.newArrayList();
		List<Labels> sentenceLabels = Generics.newArrayList();

		if(trees != null){
			for (Tree tree : trees){
				CoreLabel label  = (CoreLabel)tree.label();
				verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));			
				sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
			}
		}

		//		List<Integer> trueLabels = Generics.newArrayList();
		//		List<Integer>	predictedLabels = Generics.newArrayList();
		List<SimpleMatrix> nodevectors = Generics.newArrayList();
		List<String> phrases = Generics.newArrayList();
		for (int i=0; i<trees.size(); i++){	
			if( trees.get(i).isLeaf())
				continue;
			//			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
			Tree tree = trees.get(i); //sentence tree
			//				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
			//				List<Tree> leaves = tree.getLeaves();
			//				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
			//					continue;  //sentence tokens and labels mismatch.. have to fix this
			Tree trainingTree = tree.deepCopy();				
			//				for(int wid=0; wid<leaves.size(); wid++){					
			//					this.cag.setVerbNWordIndexFeatures(trainingTree, verbIndex, wid);
			this.cag.forwardPropagateTree(trainingTree); //calculate nodevectors

			List<Tree> phraseTrees = getPhraseTrees(trainingTree, 0, 15);
			for(Tree phraseTree : phraseTrees){

				nodevectors.add(RNNCoreAnnotations.getNodeVector(phraseTree));
				phrases.add(getPhrase(phraseTree));

			}
			//				}
			//			}

		}

		String vecFileName = "data/results/vectors_srl.txt";
		String phrasesFileName = "data/results/phrases_srl.txt";
		writePhrasesNVectors(nodevectors, phrases, vecFileName, phrasesFileName);
	}


	private void writePhrasesNVectors(List<SimpleMatrix> nodevectors, List<String> phrases){
		PrintWriter vecWriter = null;
		PrintWriter phWriter = null;
		try {

			vecWriter = new PrintWriter(Config.PROJECT_HOME+"data/results/vectors_rae_wiki.txt", "UTF-8");
			phWriter = new PrintWriter(Config.PROJECT_HOME+"data/results/phrases_rae_wiki.txt", "UTF-8");
			for(SimpleMatrix vector : nodevectors){ 
				//				String catlabel = categories.get(predLabel);
				String vecRow = "";
				for(int i=0; i < vector.numRows(); i++){
					Double value = vector.get(i, 0);
					vecRow +=value.toString()+" ";
				}
				vecWriter.println(vecRow);
			}

			for(String phrase : phrases)
				phWriter.println(phrase);

			vecWriter.close();
			phWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	private void writePhrasesNVectors(List<SimpleMatrix> nodevectors, List<String> phrases, String vecFileName, String phrasesFileName){
		PrintWriter vecWriter = null;
		PrintWriter phWriter = null;
		try {

			vecWriter = new PrintWriter(vecFileName, "UTF-8");
			phWriter = new PrintWriter(phrasesFileName, "UTF-8");
			for(SimpleMatrix vector : nodevectors){ 
				//				String catlabel = categories.get(predLabel);
				String vecRow = "";
				for(int i=0; i < vector.numRows(); i++){
					Double value = vector.get(i, 0);
					vecRow +=value.toString()+" ";
				}
				vecWriter.println(vecRow);
			}

			for(String phrase : phrases)
				phWriter.println(phrase);

			vecWriter.close();
			phWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	private String getPhrase(Tree subtree){

		List<Tree> subtreeleaves = subtree.getLeaves();
		String phrase = "";
		for(Tree leaf : subtreeleaves){
			phrase += leaf.label().value()+" ";
		}
		return phrase;
	}

	public void create_vectors_file2(List<Tree> trees){
		System.out.println("Creating phrases and vectors files...");
		Categories iobCategories = Categories.getIOBCategories();


		List<VerbIds> verbIndices = Generics.newArrayList();
		List<Labels> sentenceLabels = Generics.newArrayList();

		HashMap<Integer, String> revMap = new HashMap<>();
		for(String key : iobCategories.keySet()){
			revMap.put(iobCategories.get(key), key);
		}

		List<SimpleMatrix> nodevectors = Generics.newArrayList();
		List<String> phrases = Generics.newArrayList();		

		if(trees != null){
			for (Tree tree : trees){
				CoreLabel label  = (CoreLabel)tree.label();
				verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));			
				sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
			}
		}

		for (int i=0; i<trees.size(); i++){	
			if((verbIndices.get(i).size() == 0) || trees.get(i).isLeaf())
				continue;
			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
				Tree tree = trees.get(i); //sentence tree
				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
				List<Tree> leaves = tree.getLeaves();
				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
					continue;  //sentence tokens and labels mismatch.. have to fix this

				Tree trainingTree = tree.deepCopy();
				//				this.cag2.setVerbNWordIndexFeatures(trainingTree, verbIndex);
				this.cag2.setAdditionalFeatures(trainingTree, verbIndex);
				List<Integer> trueLabels = sentenceLabels.get(i).get(nverbs);
				Tree verbNode = trainingTree.getLeaves().get(verbIndex);
				this.cag2.forwardPropagateTree(trainingTree, trainingTree, verbNode); //calculate nodevectors

				//get phrase and its vector
				for(int i1=0; i1< trueLabels.size(); i1++){
					if(revMap.get(trueLabels.get(i1)).startsWith("B-") ){
						int endPos = i1+1;
						while((endPos < trueLabels.size()) && revMap.get(trueLabels.get(endPos)).startsWith("I-"))
							endPos += 1;


						Tree subtree = this.cag2.findSubTree(trainingTree, i1, endPos-1);
						if(subtree.getLeaves().size() >=2 && subtree.getLeaves().size() <=10){
							//							nodevectors.add(RNNCoreAnnotations.getNodeVector(subtree));
							nodevectors.add(RNNCoreAnnotations.getRAEVector(subtree));
							phrases.add(getPhrase(subtree));
						}

					}			
				}
			}
		}

		writePhrasesNVectors(nodevectors, phrases);
	}

	public void create_vectors_fileRAE(List<Tree> trees){
		System.out.println("creating phrases and vectors ... RAE..");

		List<SimpleMatrix> nodevectors = Generics.newArrayList();
		List<String> phrases = Generics.newArrayList();		


		for (int i=0; i<trees.size(); i++){	
			if(trees.get(i).isLeaf())
				continue;

			Tree tree = trees.get(i); //sentence tree



			Tree trainingTree = tree.deepCopy();

			try{
				this.cag3.forwardPropagateTree(trainingTree); //calculate nodevectors
			}catch(AssertionError e){
				//						System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
				continue;
			}

			List<Tree> phraseTrees = getPhraseTrees(trainingTree, 0, 15);
			for(Tree pTree : phraseTrees){				
				//				attachRecError(trainingTree);
				phrases.add(getPhrase(pTree));
				nodevectors.add(RNNCoreAnnotations.getNodeVector(pTree));

				//				if(phrases.size()%saveIter == 0){
				//					writePhrasesNVectors(nodevectors, phrases);
				//					phrases.clear();
				//					nodevectors.clear();

				//				}
			}




		}
		String vecFileName = "data/results/vectors_urae.txt";
		String phrasesFileName = "data/results/phrases_urae.txt";
		writePhrasesNVectors(nodevectors, phrases, vecFileName, phrasesFileName);




	}


	public void create_vectors_fileRAEContext(List<Tree> trees){
		System.out.println("creating phrases and vectors ... RAE..");

		List<SimpleMatrix> nodevectors = Generics.newArrayList();
		List<String> phrases = Generics.newArrayList();		

		


		int saveIter = 50000;

		for (int i=0; i<trees.size(); i++){	
			if(trees.get(i).isLeaf())
				continue;

			Tree tree = trees.get(i).deepCopy(); //sentence tree
			
			try{
				this.cag3.forwardPropagateTree(tree); //calculate nodevectors
			}catch(AssertionError e){
				//						System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
				continue;
			}
			List<Tree> phraseTrees = getPhraseTrees(tree, 0, 15);
			for(Tree pTree : phraseTrees){				
				phrases.add(getPhrase(pTree));
				nodevectors.add(RNNCoreAnnotations.getNodeVector(pTree));

//				if(phrases.size()%saveIter == 0){
//					//					writePhrasesNVectors(nodevectors, phrases);
//					for(SimpleMatrix vector : nodevectors){ 
//						//				String catlabel = categories.get(predLabel);
//						String vecRow = "";
//						for(int j=0; j < vector.numRows(); j++){
//							Double value = vector.get(j, 0);
//							vecRow +=value.toString()+" ";
//						}
//						vecWriter.println(vecRow);
//					}
//					for(String phrase : phrases)
//						phWriter.println(phrase);
//
//					phrases.clear();
//					nodevectors.clear();
//				}

			}
		}
		String vecFileName = "data/results/vectors_urae-cs.txt";
		String phrasesFileName = "data/results/phrases_urae-cs.txt";
		writePhrasesNVectors(nodevectors, phrases, vecFileName, phrasesFileName);
		

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

	public void create_words_vectorFile(String treePath){

		List<Tree> trees = SRLUtils.readTrees(treePath);
		//		Set<String> vocab = getRelevantVocab(trees, this.model.wordVectors);
		Set<String> vocab = this.model.wordVectors.keySet();
		List<SimpleMatrix> vectors = Generics.newArrayList();
		List<String> words = Generics.newArrayList();

		for(String word:vocab){
			words.add(word);
			vectors.add(this.model.getWordVector(word));
		}

		String vecFileName = "data/results/vectors_srl_vocab.txt";
		String phrasesFileName = "data/results/words_srl_vocab.txt";
		writePhrasesNVectors(vectors, words, vecFileName, phrasesFileName );
	}

	public void create_rAvgFile(String treePath){
		List<Tree> trees = SRLUtils.readTrees(treePath);
		List<SimpleMatrix> nodevectors = Generics.newArrayList();
		List<String> phrases = Generics.newArrayList();
		for (int i=0; i<trees.size(); i++){	
			if( trees.get(i).isLeaf())
				continue;
			//			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
			Tree tree = trees.get(i); //sentence tree
			//				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
			//				List<Tree> leaves = tree.getLeaves();
			//				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
			//					continue;  //sentence tokens and labels mismatch.. have to fix this
			Tree trainingTree = tree.deepCopy();				
			//				for(int wid=0; wid<leaves.size(); wid++){					
			//					this.cag.setVerbNWordIndexFeatures(trainingTree, verbIndex, wid);
			rAvg_forwardPropagate(trainingTree); //calculate nodevectors

			List<Tree> phraseTrees = getPhraseTrees(trainingTree, 0, 15);
			for(Tree phraseTree : phraseTrees){

				nodevectors.add(RNNCoreAnnotations.getNodeVector(phraseTree));
				phrases.add(getPhrase(phraseTree));

			}
			//				}
			//			}

		}

		String vecFileName = "data/results/vectors_ravg.txt";
		String phrasesFileName = "data/results/phrases_ravg.txt";
		writePhrasesNVectors(nodevectors, phrases, vecFileName, phrasesFileName);

	}

	public void rAvg_forwardPropagate(Tree tree){
		SimpleMatrix nodeVector = null;
		if (tree.isLeaf()) {			
			String word = tree.label().value();	
			if(word == null)
				throw new AssertionError("Tree containing a null word");
			nodeVector = model.getWordVector(word);	

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			rAvg_forwardPropagate(tree.children()[0]);
			rAvg_forwardPropagate(tree.children()[1]);			

			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			//			SimpleMatrix childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector);

			nodeVector = 	leftVector.plus(rightVector).scale(0.5);		

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}

		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);
	}

	public static void main(String[] args) {
		String modelPath = args[0];
		String treePath = args[1];

		List<Tree> trees = SRLUtils.readTreesWithGoldLabels(treePath, Config.VERBIDS_TEST, Config.LABELS_TEST, Config.POS_TEST);
//		List<Tree> allTrees = SRLUtils.readTrees(treePath);
		RNTNModel model = RNTNModel.loadSerialized(modelPath);
//		RNNPhraseModel rnnPhModel = RNNPhraseModel.loadSerialized(modelPath);

		Evaluate eval = new Evaluate(model);
		//		eval.create_rAvgFile(treePath);
		//		eval.create_words_vectorFile(treePath);
		//		eval.create_vectors_file(allTrees);
				eval.eval(trees);
		//		eval.create_preds_file(trees);
		//		eval.printSummary();
		//				eval.create_vectors_file2(trainTrees);

//				Evaluate eval = new Evaluate(rnnPhModel, allTrees, 0); //for reconRankingcost				
//				double cost = eval.eval_reconRanking();
				
//		Evaluate eval = new Evaluate(rnnPhModel);
//		eval.create_vectors_fileRAE(allTrees);
//		eval.create_vectors_fileRAEContext(allTrees);
		//		eval.eval2(trees);
		//		eval.printSummary();
		//		String trainedRAEModelName = "data/models/srl/rae_uf_cost333.mo-0028";
		//		double score = eval.eval_compatibility(trainTrees.subList(0, 100), trainedRAEModelName);
		//		System.out.println(score);


	}
}
