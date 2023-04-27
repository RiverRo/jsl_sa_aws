import logging
from sanic import Sanic
from sanic.response import json, text
from sanic.log import logger
from sanic.exceptions import NotFound, ServerError

from johnsnowlabs import finance, nlp
# from johnsnowlabs import *

app = Sanic('jsl_sa_aws')
app.update_config('./config.py')      # add config.py key=value to app
logging.basicConfig(filename=app.config.LOG_FILENAME)

# spark = start_spark()
spark = nlp.start()


async def run_sa(news:str):
    """Run sentiment analysis"""
    news = [[news]]
    document_assembler = nlp.DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')
    tokenizer = nlp.Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')
    # ERROR IN THE FOLLOWING LINE
    classifier = finance.BertForSequenceClassification.pretrained('finclf_bert_news_tweets_sentiment_analysis', 'en', 'finance/models')\
        .setInputCols(['document','token'])\
        .setOutputCol('class_')
    pipeline = nlp.Pipeline(stages = [document_assembler, tokenizer, classifier])
    data = spark.createDataFrame(news).toDF("text")
    df = pipeline.fit(data).transform(data)
    logger.info(f'[RUN_SA df.columns] {df.columns}')
    logger.info(f'[RUN_SA df.show()] {df.show()}')
    return {} 


@app.route("/", methods=['GET', 'POST'])
async def sa(request):
    if request.method == 'GET':
        return text('Whatcha Looking For?')
    elif request.method == 'POST':
        news = request.form.get('news')
        sa_results = await run_sa(news)
        return json({'sa_results': sa_results})


@app.exception(NotFound)
async def handle_not_found(request, exception):
    return text("Sorry, that page does not exist.", status=404)


@app.exception(ServerError)
async def handle_server_error(request, exception):
    return text("Oops! Something went wrong.", status=500)

if __name__ == '__main__':
    app.run(dev=True, workers=2, host=app.config.HOST, port=app.config.PORT)  #HOST="0.0.0.0"  PORT=5002