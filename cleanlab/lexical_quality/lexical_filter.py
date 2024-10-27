import pandas as pd
import os
import zipfile
from sklearn.metrics import accuracy_score, pairwise
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab.classification import CleanLearning

import language_tool_python
import textstat

def assess_lexical_quality(text):
    metrics = {}
    tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org')

    #Grammar check
    matches = tool.check(text)
    metrics['grammar_quality'] = 1 - len(matches) / len(text.split()) if text.split() else 1

    #Spelling check 
    words = text.split()
    misspelled = [match.context[match.offset:match.offset + match.errorLength] for match in matches if match.ruleId == 'MORFOLOGIK_RULE_EN_US']
    metrics['spelling_accuracy'] = 1 - len(misspelled) / len(words) if words else 1

    # Readability - measures how easy the text is to read by analysing sentence length and word complexity
    metrics['readability'] = textstat.flesch_reading_ease(text) 

    #Complexity - uses the Flesch-Kincaid Grade level to determine complexity equivalent to grade level of education
    metrics['complexity'] = textstat.flesch_kincaid_grade(text)

    # Coherence (complex implementation using huggingface transformer)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    coherence_scores = pairwise.cosine_similarity(embeddings)
    metrics['complex_coherence'] = coherence_scores.mean()
    
    # Coherence (basic implementation)
    simple_coherence_score = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
    metrics['simple_coherence'] = simple_coherence_score

    print(metrics)

    return metrics

def filter_by_lexical_quality(metrics):
    # Determine using the lexical quality metrics if they indicate an issue with the lexical quality that could affect the labelling

    poor = 0
    moderate = 0
    good = 0
    flag_label_issue = False

    #Grammar

    if metrics["grammar_quality"] <= 0.3:
        poor += 1
    elif metrics["grammar_quality"] > 0.3 and metrics["grammar_quality"] <=0.7:
        moderate += 1
    else:
        good +=1 

    #Spelling

    if metrics["spelling_accuracy"] <= 0.3:
        poor += 1
    elif metrics["spelling_accuracy"] > 0.3 and metrics["spelling_accuracy"] <=0.7:
        moderate += 1
    else:
        good +=1 

    #Readbility

    if metrics["readability"] <= 30:
        poor += 1
    elif metrics["readability"] > 30 and metrics["readability"] <=70:
        moderate += 1
    else:
        good +=1 

    #Complexity

    if metrics["complexity"] <= 3:
        poor += 1
    elif metrics["complexity"] > 3 and metrics["compexity"] <10:
        moderate += 1
    else:
        good +=1 

    #Complex Coherence

    if metrics["complex_coherence"] <= 0.3:
        poor += 1
    elif metrics["complex_coherence"] > 0.3 and metrics["complex_coherence"] <=0.7:
        moderate += 1
    else:
        good +=1 

    #Simple Coherence

    if metrics["simple_coherence"] <= 30:
        poor += 1
    elif metrics["simple_coherence"] > 30 and metrics["simple_coherence"] <=70:
        moderate += 1
    else:
        good +=1 

    if (poor > good) or (poor >=2) or (good <=2):
        flag_label_issue = True

    return flag_label_issue 

def lexical_quality(texts):

    #Create a dataframe containing the text, the corresponding metrics, and a boolean indicating if there is likely to be an issue before applying in the CL algorithm.

    metrics_list = []

    for text in texts:
        metrics = assess_lexical_quality(text)
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['text'] = texts #Add original texts as a column
    df_metrics['flag_label_issue'] = df_metrics.apply(filter_by_lexical_quality, axis=1)

    return df_metrics 

def extract_and_read_txt_files(zip_path):
    # Create a directory to extract files
    extraction_path = 'extracted_files'
    os.makedirs(extraction_path, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # List to store the contents of the text files
    text_contents = []

    # Read each .txt file and convert to string
    for filename in os.listdir(extraction_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(extraction_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_contents.append(file.read())

    return text_contents


#def testing_local():
 #   texts = extract_and_read_txt_files('cleanlab/lexical_quality/example_dataset/example.zip')
  #  lexical_quality(texts)





