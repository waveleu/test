# coding: utf-8

from pyspark import SparkConf
from pyspark.sql import SparkSession
from sklearn.externals import joblib
from datetime import timedelta, date
from absl import app
from absl import flags

def get_day_of_day(n=0):
    '''''
    if n>=0,date is larger than today
    if n<0,date is less than today
    date format = "YYYY-MM-DD"
    '''
    if (n < 0):
        n = abs(n)
        return date.today() - timedelta(days=n)
    else:
        return date.today() + timedelta(days=n)


def check_meid(argv):
    # 根据门店，天数取数据，利用模型文件去验证概率
    meid = FLAGS.meid
    if meid == None:
        return None, None
    n = FLAGS.n
    print("================================")
    D = []
    for i in range(n):
        D.append("dt='" + str(get_day_of_day(-i)) + "'")
    D = " or ".join(D)
    print(D)

    # 配置spark客户端
    conf = SparkConf().setAppName("xgboost_test")

    spark = SparkSession \
        .builder \
        .config(conf=conf) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext

    shop = spark.sql("select meid, sum(amount) as sum_pay, count(amount) as num_pay, avg(amount) as avg_pay, percentile_approx(amount, 0.5) as medin_pay, \
    max(amount) as max_pay, min(amount) as min_pay, variance(amount) as variance_pay, var_samp(amount) as var_samp_pay, stddev_pop(amount) as stdpop_pay, \
    stddev_samp(amount) as std_samp_pay, avg(pmod(cast(datediff(to_date(created_at),'2017-07-02') as int),7)) as weekday, avg(cast(hour(created_at) as int)) as hour, \
    sum(case trade_type when 'pay.weixin.jspay' then 1 else 0 end) as pay_wei_js, sum(case trade_type when 'pay.alipay.jspay' then 1 else 0 end) as pay_ali_js, \
    sum(case trade_type when 'pay.weixin.micro' then 1 else 0 end) as pay_wei_micro, sum(case trade_type when 'pay.ali.micro' then 1 else 0 end) as pay_ali_micro, sum(case when bank_type like '%CFT%' then 1 else 0 end) as cft_num, \
    sum(case when bank_type like '%PCREDIT%' then 1 else 0 end) as pcredit_num, sum(case when bank_type like '%ALIPAYACCOUNT%' then 1 else 0 end) as alipayccount_num, \
    sum(case when bank_type like '%BANKCARD%' then 1 else 0 end) as bankcard_num, sum(case when bank_type like '%COUPON%' then 1 else 0 end) as coupon_num, \
    sum(case when bank_type like '%DEBIT%' then 1 else 0 end) as debit_num, sum(case when bank_type like '%CREDIT%' then 1 else 0 end) as credit_num, \
    sum(case when bank_type like '%LQT%' then 1 else 0 end) as lqt_num, sum(case when bank_type like '%DEPOSIT%' then 1 else 0 end) as deposit_num  from (select * from lehuipay.payments where status='success' and  (" + D + ") and meid=" + str(meid) + ") group by meid")

    clf = joblib.load("./xg_model.pkl")
    shop = shop.toPandas()
    print(shop)
    shop = shop.drop(['meid'], axis=1)
    return clf.predict_proba(shop)

FLAGS = flags.FLAGS
flags.DEFINE_integer('meid', None, 'shop ID')
flags.DEFINE_integer('n', 2, 'the days num of data')
if __name__ == "__main__":
    # execute only if run as a script
    app.run(check_meid)
