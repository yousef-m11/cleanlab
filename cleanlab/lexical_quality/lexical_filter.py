import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score, pairwise
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab.classification import CleanLearning

import language_tool_python
import textstat

text = "I hate wet and reiny days. It rained a lot in 1816.... a lot - like everyday; the weather in Europe was abnormally wet because it rained in Switzerland on 130 out of the 183 days from April to September. If I was Mary Shelley I might decide to write a book too, or at least some social media posts. Afterall, it was the onnly thing you could do without cellphones or TV or anything. Sounds sooooo boring! She said that she passed the summer of 1816 in the environs of Geneva...we occasionally amused ourselves with some German stories of ghosts... These tales excited in us a playful desire of imitation. So, people were stuck inside and bored. Mary Shelley decided to write a book becuase it was so awful outside. I can totally see her point, you know? I guess I would go crazy if there was nothing else to do besides look out at the rain n read."
text2 = "The weather in 1816 Europe was abnormally wet, keeping many inhabitants indoors that summer. From April until September of that year, \"it rained in Switzerland on 130 out of the 183 days from April to September\" (Phillips, 2006). Unlike today, one could not simply turn on a television or swipe around the Internet, looking at posts and videos in order to entertain oneself. Instead, it was much more common for the educated people of the day to spend time reading, discussing well-known authors and artists of the day, playing at cards and walking in their gardens and walking paths. If you were Mary Shelley in the company of Byron and others, you amused each other by reading out loud, sharing a common interest in a particular book, and sharing with the others your own writing. In her introduction to Frankenstein, her explanation of how this extraordinary novel came to be was due, at least in part, to the weather and the company (Shelley, 1816).  \"I passed the summer of 1816 in the environs of Geneva. The season was cold and rainy, and ...we occasionally amused ourselves with some German stories of ghosts... These tales excited in us a playful desire of imitation\" (Shelley, as quoted in Phillips, 2006)."
text3 = "I speaking English very best."
text4 = "In the heart of a bustling city, where the sounds of honking cars and lively chatter filled the air, a small café nestled between towering skyscrapers offered a sanctuary for those seeking respite. The aroma of freshly brewed coffee mingled with the sweet scent of pastries, inviting passersby to step inside. As sunlight streamed through the large windows, casting a warm glow on the rustic wooden tables, patrons engaged in quiet conversations or immersed themselves in books. Outside, the vibrant energy of urban life continued, but within the café, time seemed to slow down, creating a perfect atmosphere for reflection and connection. Each sip of coffee became a moment to savor, a brief escape from the fast pace of the world outside."

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
    #metrics['easily_fixable_mistakes']

    # Readability - measures how easy the text is to read by analysing sentence length and word complexity
    metrics['readability'] = textstat.flesch_reading_ease(text) 

    #Complexity - uses the Flesch-Kincaid Grade level to determine complexity equivalent to grade level of education
    metrics['complexity'] = textstat.flesch_kincaid_grade(text)

    # Coherence (basic implementation)
    # You might want to enhance this based on your needs.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    coherence_scores = pairwise.cosine_similarity(embeddings)
    metrics['Complex coherence'] = coherence_scores.mean()
    
    simple_coherence_score = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
    metrics['Simple coherence'] = simple_coherence_score

    return metrics

#texts = [text, text2, text3]
metrics = assess_lexical_quality(text4)
print(metrics)
