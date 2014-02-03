package iais.io;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.PTBEscapingProcessor;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.Generics;

/**
 * Reads the SRL dataset and writes it to the appropriate files.
 *
 * @author Bhanu Pratap
 */
public class ReadSRLDataset {
	public static Tree convertTree(List<Integer> parentPointers, List<String> sentence, 
			//		  Map<List<String>, Integer> phraseIds, Map<Integer, Double> sentimentScores, 
			PTBEscapingProcessor escaper) {
		int maxNode = 0;
		for (Integer parent : parentPointers) {
			maxNode = Math.max(maxNode, parent);
		}


		Tree[] subtrees = new Tree[maxNode + 1];
		Tree root = null;

		for (int i = 0; i < sentence.size(); ++i) {
			if(i >= subtrees.length){
				continue;
			}

			CoreLabel word = new CoreLabel();
			word.setValue(sentence.get(i));
			Tree leaf = new LabeledScoredTreeNode(word);
			subtrees[i] = new LabeledScoredTreeNode(new CoreLabel());
			subtrees[i].addChild(leaf);
		}


		for (int i = sentence.size(); i <= maxNode; ++i) {
			subtrees[i] = new LabeledScoredTreeNode(new CoreLabel());
		}

		boolean[] connected = new boolean[maxNode + 1];

		for (int index = 0; index < parentPointers.size(); ++index) {
			if (parentPointers.get(index) == -1) {
				if (root != null) {
					throw new RuntimeException("Found two roots for sentence " + sentence);
				}
				root = subtrees[index];
			} else {
				// Walk up the tree structure to make sure that leftmost
				// phrases are added first.  Otherwise, if the numbers are
				// inverted, we might get the right phrase added to a parent
				// first, resulting in "case zero in this", for example,
				// instead of "in this case zero"
				// Note that because we keep track of which ones are already
				// connected, we process this at most once per parent, so the
				// overall construction time is still efficient.
				connect(parentPointers, subtrees, connected, index);
			}
		}

		for (int i = 0; i <= maxNode; ++i) {
			List<Tree> leaves = subtrees[i].getLeaves();
			List<String> words = CollectionUtils.transformAsList(leaves, new Function<Tree, String>() { 
				public String apply(Tree tree) { return tree.label().value(); }
			});
			//			Integer phraseId = phraseIds.get(words);
			//			if (phraseId == null) {
			//				throw new RuntimeException("Could not find phrase id for phrase " + sentence);
			//			}
			//			// TODO: should we make this an option?  Perhaps we want cases
			//			// where the trees have the phrase id and not their class
			//			Double score = sentimentScores.get(phraseId);
			//			if (score == null) {
			//				throw new RuntimeException("Could not find sentiment score for phrase id " + phraseId);
			//			}
			//			// TODO: make this a numClasses option
			//			int classLabel = Math.round((float) Math.floor(score * 5.0));
			//			if (classLabel > 4) {
			//				classLabel = 4;
			//			}
			//			subtrees[i].label().setValue(Integer.toString(classLabel));
		}

		for (int i = 0; i < sentence.size(); ++i) {
			Tree leaf = subtrees[i].children()[0];
			leaf.label().setValue(escaper.escapeString(leaf.label().value()));
		}


		return root;
	}

	private static void connect(List<Integer> parentPointers, Tree[] subtrees, boolean[] connected, int index) {
		if (connected[index]) {
			return;
		}
		if (parentPointers.get(index) < 0) {
			return;
		}
		subtrees[parentPointers.get(index)].addChild(subtrees[index]);
		connected[index] = true;
		connect(parentPointers, subtrees, connected, parentPointers.get(index));
	}

	private static void writeTrees(String filename, List<Tree> trees) {
		try {
			FileOutputStream fos = new FileOutputStream(filename,true);
			BufferedWriter bout = new BufferedWriter(new OutputStreamWriter(fos));

			for (int id=0; id < trees.size() ;id++) {
				bout.write(trees.get(id).toString());
				bout.write("\n");
			}
			bout.flush();
			fos.close();
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}
	}

	/**
	 * This program converts the SRL Parent pointer trees format of the 
	 * SRL data set into into trees readable with the
	 * normal TreeReaders.
	 * <br>
	 * An example command line is
	 * <br>
	 * <code>java iais.io.ReadSRLDataset  -tokens data/corpus/rawsentences_srl.train -parse data/corpus/srl_trees.train -verbs data/corpus/srl_vids.train -out data/corpus/srlRNTN.train </code>
	 * <br>
	 * 
	 * 
	 * Each of these arguments is required.
	 */
	public static void main(String[] args) {


		String tokensFilename = null;
		String parseFilename = null;
		String outFilename = null;
		String verbsFilename = null;

		int argIndex = 0;
		while (argIndex < args.length) {
			if(args[argIndex].equalsIgnoreCase("-tokens")) {
				tokensFilename = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-parse")) {
				parseFilename = args[argIndex + 1];
				argIndex += 2;

			} else if (args[argIndex].equalsIgnoreCase("-out")) {
				outFilename = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-verbs")) {
				verbsFilename = args[argIndex + 1];
				argIndex += 2;
			}
			else {
				System.err.println("Unknown argument " + args[argIndex]);
				System.exit(2);
			}
		}

		// Sentence file is formatted
		//   w1|w2|w3...
		List<List<String>> sentences = Generics.newArrayList();
		for (String line : IOUtils.readLines(tokensFilename, "utf-8")) {
			String[] sentence = line.split("\\|");
			sentences.add(Arrays.asList(sentence));
		}



		// Read lines from the tree structure file.  This is a file of parent pointers for each tree.
		int index = 0;
		PTBEscapingProcessor escaper = new PTBEscapingProcessor();
		List<Tree> trees = Generics.newArrayList();
		int nTrees = 0; 
		int saveIter = 20000;
		for (String line : IOUtils.readLines(parseFilename)) {
			String[] pieces = line.split("\\|");
			List<Integer> parentPointers = CollectionUtils.transformAsList(Arrays.asList(pieces), new Function<String, Integer>() { 
				public Integer apply(String arg) { return Integer.valueOf(arg) - 1; }
			});
			if(sentences.get(index).size() > 4){
				Tree tree = convertTree(parentPointers, sentences.get(index), escaper);
				trees.add(tree);
				if(trees.size() % saveIter == 0){
					nTrees += saveIter;
					System.out.format("Writing trees : %d \n", nTrees);
					writeTrees(outFilename, trees);
					trees.clear();					
				}
			}
			++index;
		}
//		writeTrees(outFilename, trees);

	}
}
