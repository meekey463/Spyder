# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:59:45 2020

@author: Meekey
"""
#unstructured Test
#cleanup using NPL
#unsupervised - clustering

from flask import Flask, request, make_response, send_file
from stemming.porter2 import stem
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
import time
import zipfile
from flasgger import Swagger

app = Flask('_name_')
swagger = Swagger(app)

def clean_text(text):
    if text:
        #whitespaces
        # a = 'this product is       really good'
        clean = ' '.join(text.split())
        red_text = [stem(word) for word in clean.split()]
        return ' '.join(red_text)

    else:
        return text
    
@app.route('/cluster', methods=['POST'])
def cluster():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses: 
        200:
            description: success
    """

#    data = pd.read_csv(request.files['dataset'])
    data = pd.read_csv(request.files.get("input_file"))

#    data = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\Spyder Code\\cluster_text.csv")
    unstructure = 'rev'
    if 'col' in request.args:
        unstructure = request.args.get('col')
    no_of_clusters = 2
    if 'no_of_clusters' in request.args:
        no_of_clusters = int(request.args.get('no_of_clusters'))
        
    data.fillna('NULL')
    
    data['clean_sum'] = data[unstructure].apply(clean_text)
    
    vectorizer = CountVectorizer(analyzer='word',
                                 stop_words='english')
    counts = vectorizer.fit_transform(data['clean_sum'])
    
    
    
    
    
    kmeans = KMeans(n_clusters = no_of_clusters)
    data['cluster_num'] = kmeans.fit_predict(counts)
    data = data.drop(['clean_sum'], axis = 1)
    # return 'this works'
    
    
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Clusters',
                  encoding = 'utf-8', index=False)
    
    clusters = []
    for i in range(np.shape(kmeans.cluster_centers_)[0]):
        data_cluster = pd.concat([pd.Series(vectorizer.get_feature_names()),
                                  pd.DataFrame(kmeans.cluster_centers_[i])], axis = 1)
        data_cluster.columns = ['keywords', 'weights']
        data_cluster = data_cluster.sort_values(['weights'], ascending=False)
        data_clust = data_cluster.head(10)['keywords'].tolist()
        clusters.append(data_clust)
    pd.DataFrame(clusters).to_excel(writer,sheet_name='Top_Keywords', encoding = 'utf-8')        
    
    #Pivot
    data_pivot = data.groupby(['cluster_num'], as_index=False).size()
    data_pivot.name = 'size'
    data_pivot = data_pivot.reset_index()
    data_pivot.to_excel(writer, sheet_name='Cluster_Report',
              encoding = 'utf-8', index=False)

    #insert a chart
    workbook = writer.book
    worksheet = writer.sheets['Cluster_Report']
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({
        'values': '=Cluster_Report!$B$2:B'+str(no_of_clusters+1)
        })
    worksheet.insert_chart('F2', chart)
    
    writer.save()
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['cluster_output.xlsx']
        files = [output]
        for i in range(len(files)):
            data = zipfile.ZipInfo(names[i])
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, files[i].getvalue())
    memory_file.seek(0)
    # return 'this works'
        
    # response = make_response(send_file(memory_file, as_attachment=True,
    #                                     attachment_filename='cluster_output.zip'))
    response = make_response(send_file(memory_file, attachment_filename='cluster_output.zip',
                                    as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment;filename=cluster_output.zip'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    


    
    
    
    
    
    
