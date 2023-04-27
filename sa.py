from johnsnowlabs import *

import json
import os

# spark = start_spark()
# spark = nlp.start()

def get_model(mtype):
    """Return a model pipeline
    
    Parameters
    ----------
    mtype: str
    
    Returns
    -------
    pipeline
    """
    pipeline = None
    document_assembler = nlp.DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')
    tokenizer = nlp.Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')
    embeddings = None
    classifier = None
    if mtype == 'a':
        # September 2022 - finclf_bert_sentiment_phrasebank
        # Financial PhraseBank by Malo et al. (2014) and
        # in-house JSL documents and annotations have been used for fine-tuning
        # sequenceClassifier_loaded = nlp.BertForSequenceClassification.pretrained('finclf_bert_sentiment_phrasebank', 'en', 'finance/models') \
        classifier = finance.BertForSequenceClassification.pretrained('finclf_bert_sentiment_phrasebank', 'en', 'finance/models') \
            .setInputCols(['document','token']) \
            .setOutputCol('class_')
    elif mtype == 'b':
        # September 2022 - finclf_bert_sentiment
        # Superior performance on financial tone analysis task
        # Fine-tuned model on 12K+ manually annotated analyst reports on top of Financial Bert Embeddings
        classifier = finance.BertForSequenceClassification.pretrained("finclf_bert_sentiment", "en", "finance/models")\
            .setInputCols(['document','token'])\
            .setOutputCol('class_')
    elif mtype == 'c':
        # March 2023 - finclf_bert_news_tweets_sentiment_analysis
        # Financial news articles and tweets that have been labeled with three different classes: Bullish, Bearish and Neutral. 
        # Trained data covers a wide range of financial topics including stocks, bonds, currencies, and commodities
        classifier = finance.BertForSequenceClassification.pretrained('finclf_bert_news_tweets_sentiment_analysis', 'en', 'finance/models')\
            .setInputCols(['document','token'])\
            .setOutputCol('class_')
    elif mtype == 'd':
        # August 2022 - finclf_distilroberta_sentiment_analysis
        # In-house financial documents and Financial PhraseBank by Malo et al. (2014)
        classifier = nlp.RoBertaForSequenceClassification.pretrained('finclf_distilroberta_sentiment_analysis','en', 'finance/models') \
            .setInputCols(['document', 'token']) \
            .setOutputCol('class_')
    elif mtype == 'e':
        # November 2022 - finclf_auditor_sentiment_analysis
        embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")
        classifier =  nlp.ClassifierDLModel.pretrained("finclf_auditor_sentiment_analysis", "en", "finance/models") \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("class_")
        
    if not embeddings:
        pipeline = nlp.Pipeline(stages = [document_assembler, tokenizer, classifier])
    else:
        pipeline = nlp.Pipeline(stages = [document_assembler, embeddings, classifier])
    return pipeline



# def get_model(mtype):
#     """Return a model pipeline
    
#     Parameters
#     ----------
#     mtype: str
    
#     Returns
#     -------
#     pipeline
#     """
#     pipeline = None
#     document_assembler = nlp.DocumentAssembler() \
#         .setInputCol('text') \
#         .setOutputCol('document')
#     tokenizer = nlp.Tokenizer() \
#         .setInputCols(['document']) \
#         .setOutputCol('token')
#     embeddings = None
#     classifier = None
#     if mtype == 'a':
#         # September 2022 - finclf_bert_sentiment_phrasebank
#         # Financial PhraseBank by Malo et al. (2014) and
#         # in-house JSL documents and annotations have been used for fine-tuning
#         # sequenceClassifier_loaded = nlp.BertForSequenceClassification.pretrained('finclf_bert_sentiment_phrasebank', 'en', 'finance/models') \
#         classifier = finance.BertForSequenceClassification.pretrained('finclf_bert_sentiment_phrasebank', 'en', 'finance/models') \
#             .setInputCols(['document','token']) \
#             .setOutputCol('class_')
#     elif mtype == 'b':
#         # September 2022 - finclf_bert_sentiment
#         # Superior performance on financial tone analysis task
#         # Fine-tuned model on 12K+ manually annotated analyst reports on top of Financial Bert Embeddings
#         classifier = finance.BertForSequenceClassification.pretrained("finclf_bert_sentiment", "en", "finance/models")\
#             .setInputCols(['document','token'])\
#             .setOutputCol('class_')
#     elif mtype == 'c':
#         # March 2023 - finclf_bert_news_tweets_sentiment_analysis
#         # Financial news articles and tweets that have been labeled with three different classes: Bullish, Bearish and Neutral. 
#         # Trained data covers a wide range of financial topics including stocks, bonds, currencies, and commodities
#         classifier = finance.BertForSequenceClassification.pretrained('finclf_bert_news_tweets_sentiment_analysis', 'en', 'finance/models')\
#             .setInputCols(['document','token'])\
#             .setOutputCol('class_')
#     elif mtype == 'd':
#         # August 2022 - finclf_distilroberta_sentiment_analysis
#         # In-house financial documents and Financial PhraseBank by Malo et al. (2014)
#         classifier = nlp.RoBertaForSequenceClassification.pretrained('finclf_distilroberta_sentiment_analysis','en', 'finance/models') \
#             .setInputCols(['document', 'token']) \
#             .setOutputCol('class')
#     elif mtype == 'e':
#         # November 2022 - finclf_auditor_sentiment_analysis
#         embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
#             .setInputCols("document") \
#             .setOutputCol("sentence_embeddings")
#         classifier =  nlp.ClassifierDLModel.pretrained("finclf_auditor_sentiment_analysis", "en", "finance/models") \
#             .setInputCols("sentence_embeddings") \
#             .setOutputCol("class_")
        
#     if not embeddings:
#         pipeline = nlp.Pipeline(stages = [document_assembler, tokenizer, classifier])
#     else:
#         pipeline = nlp.Pipeline(stages = [document_assembler, embeddings, classifier])
#     return pipeline
