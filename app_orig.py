import logging
import sparknlp

from sanic import Sanic
from sanic.response import json, text
from sanic.log import logger
from sanic.exceptions import NotFound, ServerError
from johnsnowlabs import *

from sa import get_model 
# from sklearn.metrics import accuracy_score, classification_report

app = Sanic('jsl_sa_aws')
app.update_config('./config.py')      # add config.py key=value to app
logging.basicConfig(filename=app.config.LOG_FILENAME)

@app.after_server_start
async def init(app):
    app.ctx.spark = start_spark()
    # logger.info(f'Running Sparknlp version: {sparknlp.version()}. Apache Spark version: {app.ctx.spark.version}')


#     sa_direction3, sa_value3 = self.sentiment_analyzer3(news)
#     content['sa_direction3'] = sa_direction3
#     content['sa_value3'] = sa_value3


async def run_sa(app, news, mtype):
    """Return sentiment analysis results
    
    Parameters
    ----------
    news: str
    mtype: str
    
    Returns
    -------
    dict
    """
    news = [[news]]
    data = app.ctx.spark.createDataFrame(news).toDF("text")
    pipeline = get_model(mtype)
    df = pipeline.fit(data).transform(data)
    logger.info(f'[RUN_SA df.columns] {df.columns}')
    logger.info(f'[RUN_SA df.show()] {df.show()}')
    return {} # [name, direction, value]??


@app.route("/", methods=['GET', 'POST'])
async def sa(request):
    if request.method == 'GET':
        return text('Whatcha Looking For?')
    elif request.method == 'POST':
        news = request.form.get('news')
        mtype = request.form.get('mtype')
        sa_results = await run_sa(app, news, mtype)
        return json({'sa_results': sa_results})


@app.exception(NotFound)
async def handle_not_found(request, exception):
    return text("Sorry, that page does not exist.", status=404)


@app.exception(ServerError)
async def handle_server_error(request, exception):
    return text("Oops! Something went wrong.", status=500)

if __name__ == '__main__':
    # app.run(host=app.config.HOST, port=app.config.PORT)
    app.run(dev=True, workers=2, host=app.config.HOST, port=app.config.PORT)