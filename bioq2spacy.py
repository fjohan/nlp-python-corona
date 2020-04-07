# 22:17 20-03-23

#%%
import codecs
import spacy
import urllib.request

from bs4 import BeautifulSoup

from spacy import displacy
from spacy.tokens import Span
from spacy_lookup import Entity


#%%
nlp = spacy.load("en")         
#%%
#with codecs.open("31991541.xml", "r", "utf-8") as file:
with codecs.open("a1_10.xml", "r", "utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")
#%%
myUrl = 'https://raw.githubusercontent.com/Aitslab/corona/master/manuscript/Supplemental_file4.xml'

with urllib.request.urlopen(myUrl) as response:
   html = response.read()
   soup = BeautifulSoup(html, "html.parser")
#%%
print(soup.prettify())

soup.source
soup.source.string
#%%
type_list = []
data_point_list = []
for docs in soup.find_all('document'):
    did = docs.id.string
    for psgs in docs.find_all('passage'):
        ptype = psgs.find(key="type")
        type_list.append((did, ptype.string))
        poff = int(psgs.offset.string)
        for t in psgs.find_all('text', recursive=False):
            #print('----')
            #print(t.string)
            di = {}
            ann_list = []
            for anns in psgs.find_all('annotation'):
                tp = anns.find(key="type")
                off = int(anns.location['offset'])
                lng = int(anns.location['length'])
                #print(tp.string, off, lng)
                ann_list.append((off-poff, off+lng-poff, str(tp.string)))
                #ann_list.append({'end': int(off)+int(lng),'label': tp.string, 'start': int(off)})
            di["entities"] = ann_list
            data_point = [str(t.string), di]
            data_point_list.append(data_point)
            #print(data_point)
#%%
# helper to convert char index to token index
# https://stackoverflow.com/questions/55109468/spacy-get-token-from-character-index
def get_token_for_char(doc, char_idx):
    for i, token in enumerate(doc):
        #print (i,token, char_idx, token.idx)
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            #return (i, token)
            return i 
        if char_idx < token.idx:
        #    return (i, doc[i - 1])
            return i
    return i+1
#%%
dp_index = 19
doc = nlp(data_point_list[dp_index][0])
#print(doc.text)
d=data_point_list[dp_index][1]
doc.ents = []
spans = []
for entity in d["entities"]:
    ts = get_token_for_char(doc, entity[0])
    te = get_token_for_char(doc, entity[1])
    #print(entity, ts, te)
    #print(ts,te)
    span = Span(doc, ts, te, label=entity[2])
    #print(span)
    #spans.append(span)
    doc.ents = list(doc.ents) + [span]  # add span to doc.ents
#doc.ents = spans

#print(d["entities"])

#print(doc.ents)

print(type_list[dp_index])
displacy.render(doc, jupyter = True, style = "ent")
#%%
def train_model(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(itn, losses)
    
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


#%%
import random
from pathlib import Path
TRAIN_DATA = data_point_list
train_model(None, '.', 10)
#%%
def evaluate_model(pretrained_model, test_data):
    '''evaluates model on sample of 6 examples by printing out correct entities and predicted entities by spacy model'''
    
    for test_instance in test_data:
        test_sentence, entities = test_instance        
        test_doc = pretrained_model(test_sentence)
        predicted_entities = [(ent.text, ent.label_) for ent in test_doc.ents]        
        original_entities = [(test_sentence[int(original_entity[0]): int(original_entity[1])], original_entity[2]) \
                             for original_entity in entities['entities']]
        print("\n--->" + test_sentence)
        print('predicted entities', predicted_entities)        
        print('original entities', original_entities)        
#%%
evaluate_model(nlp2, data_point_list[1:2])        
#%%
import pprint
#scorer = nlp.evaluate(docs_golds, verbose=True)
scorer = nlp2.evaluate(data_point_list[1:2], verbose=False)
pprint.pprint(scorer.scores)
#%%
# test the saved model
print("Loading from", '.')
nlp2 = spacy.load('.')
for text, _ in TRAIN_DATA[0][0]:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

#%%
import spacy
from spacy_lookup import Entity

nlp = spacy.load('en')

entity = Entity(keywords_list=['python', 'java platform'], label='ACME')
nlp.add_pipe(entity)
doc = nlp(u"I am a product manager for a java platform and python.")

[(ent.text, ent.label_) for ent in doc.ents]

#%%
from spacy.gold import GoldParse
from spacy.scorer import Scorer
def evaluate_iob(model, examples):
  scorer = Scorer()
  for input_, annot in examples:
    #print(input_)
    doc_gold_text = model.make_doc(input_)
    gold = GoldParse(doc_gold_text, entities=annot['entities'])
    pred_value = model(input_)
    scorer.score(pred_value, gold)
  #return scorer.scores
  return (pred_value, gold, scorer.scores, doc_gold_text)
#%%
test_result = evaluate_iob(nlp2, [tuple(data_point_list[1])])
print(test_result[3])
print([token.ent_iob_+'-'+token.ent_type_ if token.ent_iob_ != "O" else token.ent_iob_ for token in test_result[0]]) # pred_value
print(test_result[1].ner) # gold
test_result[2]

#%%

nlp3 = spacy.blank('en')
#entity_virus = Entity(keywords_file='SupplFile2_corona_virus.txt', label='Virus')
#nlp3.add_pipe(entity_virus)
entity_disease = Entity(keywords_file='SupplFile1_corona_disease.txt', label='Disease')
nlp3.add_pipe(entity_disease)
doc2 = nlp3(data_point_list[19][0])
[(ent.text, ent.label_) for ent in doc2.ents]


#%%
with open('/home/johanf/dev/nbsvm/MDCS_1006/MDCS_1006baretext.txt') as f:
    content = f.readlines()
nlp3 = spacy.blank('en')
#entity_virus = Entity(keywords_file='SupplFile2_corona_virus.txt', label='Virus')
#nlp3.add_pipe(entity_virus)
entity_result = Entity(keywords_file='/home/johanf/dev/nbsvm/MDCS_1006/p_result.txt', label='Result')
nlp3.add_pipe(entity_result)
#%%
text=content[3]
doc2 = nlp3(text)
print(text)
[(ent.text, ent.label_) for ent in doc2.ents]


#%%
docs = []
texts = []
for elem in soup.find_all("document"):
    for child in elem.find_all("text"):
        print(child)


import xml.etree.cElementTree as ET

tree = ET.parse('31991541.xml')
root = tree.getroot()
passages = root.findall('./document')

for passage in passages:
    passage_annotations = passage.findall('./annotation')
    passage_offset = int(passage.find('offset').text)
    passage_text = passage.find('text').text
    print(passage_text)



CLARIN Newsflash March 2020

New collections and resources available in the VLO