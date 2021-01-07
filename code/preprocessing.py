#!/usr/bin/env python
# coding: utf-8

# # load library 

# In[9]:


from subprocess import call

import re
import spacy
import tempfile
import csv
import shutil

class Preprocessing:
    def __init__(self):        
        self.nlp = spacy.load('en_core_web_md')
        
    def lemmatization(self,comment):
        lemma_comment = []
        doc = self.nlp(comment)
        for token in doc:
            lemma_word = token.lemma_
            if(lemma_word == "-PRON-"):
                lemma_word = "pron"
            lemma_comment.append(lemma_word.lower())
        comment_out = ' '.join(lemma_comment)
        comment_out = re.sub(r'[^A-Za-z0-9]+','   ', comment_out, flags=re.IGNORECASE)
        comment_out = re.sub(r'\s+',' ',comment_out)
        return comment_out
    
    def special_character_removal(self,comment):
        try:
            comment = re.sub(r'[^A-Za-z0-9.\']+',' ',comment)
            comment = re.sub(r'\s+',' ',comment)
        finally:
            return comment
    
    # create n-gram using library ngweight with default setting
    # the n-gram only create from postive class
    def create_n_gram(self,comments,labels,ngweight_folder,n_gram_file_output):
        temp_folder = tempfile.mkdtemp()
        temp = open(temp_folder + '/n_gram_preparation' ,'w')
        for index in range(len(comments)):
            if labels[index] != 0:
                temp.write(chr(0x02)+chr(0x03)+"\n")
                temp.write(comments[index]+"\n")
        temp.close()
        
        call(ngweight_folder+"/waf",shell=True)
        call(ngweight_folder+"/bin/default/ngweight -w -s 0 < "+ str(temp.name) + '>' + temp_folder + '/n_gram' ,shell=True)
        
        n_gram_filter = list()
        
        with open(temp_folder + '/n_gram') as csvfile:                    
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in reader:
                if int(row[3]) <= 1:
                    continue
                n_gram_filter.append(row)

        with open(n_gram_file_output,'w') as csvfile:
            writer = csv.writer(csvfile,delimiter='\t', quotechar='|')
            writer.writerows(n_gram_filter)
                
        # delete file
        shutil.rmtree(temp_folder)
        


# In[ ]:




