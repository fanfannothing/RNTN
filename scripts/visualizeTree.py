import sys
from nltk.tree import Tree

def test_nltk_trees(parsed_text):
    
    ''' Example of parsed_text, stanford parser output :
    
        (ROOT
  (S
    (ADVP (RB However))
    (NP
      (NP (DT the) (NNS talks))
      (, ,)
      (VP (VBN hosted)
        (PP (IN by)
          (NP (NNP Douglas) (NNP Hurd))))
      (, ,))
    (VP (VBD ended)
      (PP (IN in)
        (NP (NN stalemate))))
    (. .)))
    
    ''' 
    nltree = Tree.parse(parsed_text)
    nltree.chomsky_normal_form()
    nltree.draw()
        
if __name__ == '__main__':
    test_nltk_trees(parsed_text=sys.argv[1])