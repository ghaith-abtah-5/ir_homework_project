# ir_homework_project
To start you must call search_english() or search_arabic() in main.py
After that the selected dataset will be downloaded using ir_datasets.load() "this action may take some time", alternatively you can type "python app.py" in the root directory of the project to start the web app, next steps are mostly the same. 

After loading the dataset, its documents will be processed using process_english_text(data) or process_arabic_text(data) from text_processing.py -depeneding on the dataset you chose- these function will remove non-words and spacial charectars from every document and stop words will be removed then will toknize the document then limitize English words or stem Arabic words.

After each document finish the text processing, its tokens will be used to build the inverted_index.

When every document finish the text processing, the inverted index will be fully built. then the vector model will be generated for the documents using TfidfVectorizer from sklearn.feature_extraction.text and the transformer will be fitted to then use it for the query vector, this process will make use of vectorize_docs(documents,vectorizer) from tf_idf.py.

Note that the same process will be applied for the query except build the inverted index.

After that the candidate documents will be searched for using get_candidate_documents(query_tokens,inverted_index) making useful of query_tokens and inverted_index.

Using candidate documents we will slice the document vectors to get the candidate document vectors.

The cosine similarity will be calculated using the dot method on the vector and flatten.

The relevant document will be sorted descendingly according to their relevance score using argsort()[::-1].

Note: this step only for console program
After that we will build a dict from the shape of {query_id: {doc_id: relevance_score}} to be used with pytrec_eval library to evaluate the result.

Note: this step only for web program
The percision will be calcualted calculate_precision(related_docs,candidate_documents) from evaluation.py