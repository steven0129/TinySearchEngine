import argparse
import re
import math
import os.path
import time
from nltk.stem import PorterStemmer
from collections import defaultdict
from wsgiref.simple_server import make_server
import json
from urllib.parse import parse_qs

class Indexer:
    def __init__(self, collectionPath, indexPath, stopwordPath):
        self.index = defaultdict(list) # term --> List of (docID, [positions])
        self.documentFrequency = defaultdict(int) # term --> df
        self.totalNumOfDoc = 0
        self.collectionPath = collectionPath
        self.indexPath = indexPath
        self.stemmer = PorterStemmer()
        self.stopWords = set()
        self.removeStopWords = True
        self.punctuations = ['.', '?', '!', ',', ';', ':', '-', '—']
        self.separator = ['-', '—', '/', '_', '$', '#', '@', '&']
        with open(stopwordPath, 'r') as F:
            for line in F:
                self.stopWords.add(line.strip())

    def buildIndex(self):
        collectionFile = open(self.collectionPath, 'r')
        maxNumBuffer = 20
        docNumStr = ''
        docNum = -1
        documentText = ''

        buffer = [''] * maxNumBuffer
        insideText = False
        insideDocNo = False
        self.totalNumOfDoc = 0

        char = collectionFile.read(1)
        while char != '':  # read file until EOF
            buffer.pop(0)
            buffer.append(char)
            candidateStr = ''.join(buffer).lower()

            if candidateStr.endswith('<text>'):
                insideText = True
            elif candidateStr.endswith('</text>'):
                insideText = False
                if docNum != -1:
                    documentText = documentText[:-len('</text')]
                    for term, position in self.__preprocessIndex(documentText):
                        self.__addNewTerm(self.index, term, docNum, position)
                documentText = ''
                docNum = -1
                docNumStr = ''
                self.totalNumOfDoc += 1
            elif candidateStr.endswith('<headline>'):
                insideText = True
            elif candidateStr.endswith('</headline>'):
                documentText = documentText[:-len('</headline')]
                documentText += '\n'
                insideText = False
            elif candidateStr.endswith('<docno>'):
                insideDocNo = True
            elif candidateStr.endswith('</docno>'):
                insideDocNo = False
                docNum = int(docNumStr.lower()[:-len('</docno')].strip())
                
            elif insideDocNo:
                docNumStr += char
            elif insideText:
                documentText += char
            char = collectionFile.read(1)

        for term, docs in self.index.items():
            self.documentFrequency[term] = len(docs)

        with open(self.indexPath, 'w') as F:
            for term in sorted(self.index.keys()):
                postingStr = ''
                df = self.documentFrequency[term]
                for docID, positions in self.index[term]:
                    postingStr += f"\t{docID}: {','.join(map(str, positions))}\n"
                F.write(f"{term}:{df}\n{postingStr}\n\n")

    def loadIndex(self):
        self.index = defaultdict(list)
        with open(self.indexPath, 'r') as F:
            content = F.read()
            entries = content.strip().split('\n\n')
            for entry in entries:
                lines = entry.strip().split('\n')
                term = lines[0].split(':')[0]
                df = int(lines[0].split(':')[1])
                postings = []
                
                for line in lines[1:]:
                    if line.strip() == '':
                        continue
                    docPart = line.strip().split(':')
                    docID = int(docPart[0].strip())
                    positions = list(map(int, docPart[1].strip().split(',')))
                    postings.append((docID, positions))
                self.index[term] = postings
                self.documentFrequency[term] = df

            self.totalNumOfDoc = len({docID for postings in self.index.values() for docID, _ in postings})

    def queryWithTerm(self, queryString):
        finalDocs = set()
        tokens = self.__tokenize(queryString.strip())
        for token in tokens:
            docIDs = list(map(lambda x: x[0], self.__getDocs(token)))
            for docID in docIDs:
                finalDocs.add(docID)
        return list(finalDocs) if len(finalDocs) > 0 else []

    def queryWithTfIdf(self, queryString, topDocs=150, numSuggestedTerms=5, numPRFDocs=1):
        commands = self.__tokenize(queryString)
        docScores = defaultdict(float)

        for term in commands:
            if term in self.index:
                postings = self.index[term]
                for docID, positions in postings:
                    tf = len(positions)
                    df = self.documentFrequency[term]
                    docScores[docID] += (1 + math.log10(tf)) * math.log10(self.totalNumOfDoc / df)

        docRanked = sorted(docScores.items(), key=lambda x: x[1], reverse=True)
        suggestedTerms = self.__PRF(docRanked, numPRFDocs, numSuggestedTerms)  # Suggest terms by PRF
        return docRanked[:topDocs], suggestedTerms

    def __getDocs(self, term):
        return self.index[term]

    def __PRF(self, initDocScores, numTopInitDocs=1, numReturnTopTerms=5):
        initTopDocs = initDocScores[:numTopInitDocs]
        collectionFile = open(self.collectionPath, 'r')
        index = defaultdict(list)
        tfidf = defaultdict(float)
        maxNumBuffer = 20
        buffer = [''] * maxNumBuffer
        insideText = False
        insideDocNo = False
        docNumStr = ''
        docNum = -1
        documentText = ''

        # Create a set of document IDs to quickly lookup which ones to extract
        topDocIDs = set(list(map(lambda x: x[0], initTopDocs)))

        char = collectionFile.read(1)
        while char != '':  # Read until EOF
            buffer.pop(0)
            buffer.append(char)
            candidateStr = ''.join(buffer).lower()
            
            if candidateStr.endswith('<text>'):
                insideText = True
            elif candidateStr.endswith('</text>'):
                insideText = False
                if docNum in topDocIDs:
                    documentText = documentText[:-len('</text')]
                    for term, position in self.__preprocessIndex(documentText):
                        self.__addNewTerm(index, term, docNum, position)
                documentText = ''
                docNum = -1
                docNumStr = ''
            elif candidateStr.endswith('<headline>'):
                insideText = True
            elif candidateStr.endswith('</headline>'):
                documentText = documentText[:-len('</headline')]
                documentText += '\n'
                insideText = False
            elif candidateStr.endswith('<docno>'):
                insideDocNo = True
            elif candidateStr.endswith('</docno>'):
                insideDocNo = False
                docNum = int(docNumStr.lower()[:-len('</docno')].strip())
            elif insideDocNo:
                docNumStr += char
            elif insideText:
                documentText += char
            char = collectionFile.read(1)
        
        collectionFile.close()

        for term, postings in index.items():
            tf = sum(len(positions) for _, positions in postings)
            tfidf[term] = tf * math.log10(self.totalNumOfDoc / self.documentFrequency[term])

        return sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:numReturnTopTerms]

    def __preprocessIndex(self, sentence):
        terms = self.__tokenize(sentence)
        terms = [(term, idx + 1) for idx, term in enumerate(terms)]  # Add positions
    
        return terms

    def __tokenize(self, sentence):
        terms = []
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('\r', ' ')
        sentence = sentence.split(' ')
        for word in sentence:
            candidates = []
            if word != '' and word[-1] in self.punctuations:
                word = word[:-1]

            for sep in self.separator:
                if sep in word:
                    for w in word.split(sep):
                        if w != '' and (not w in self.stopWords or not self.removeStopWords):
                            w = self.__cleanTerm(w)
                            w = self.stemmer.stem(w)
                            candidates.append(w)

                    break

            if len(candidates) == 0:
                word = self.__cleanTerm(word)
                if word != '' and (not word in self.stopWords or not self.removeStopWords):
                    word = self.stemmer.stem(word)
                    candidates.append(word)

            terms.extend(candidates)

        return terms
    
    def __cleanTerm(self, term):
        processedTerm = ''
        for c in term:
            if c.isalnum():
                processedTerm += c

        return processedTerm

    def __addNewTerm(self, index, term, docID, position):
        postings = index[term]
        for i, (dID, positions) in enumerate(postings):
            if dID == docID:
                # Insert position in sorted order
                j = len(positions) - 1
                while j >= 0 and positions[j] > position:
                    j -= 1
                positions.insert(j + 1, position)
                postings[i] = (dID, positions)
                return

        # Insert (docID, [position]) in sorted order by docID
        insertIndex = len(postings)
        for i, (existingDocID, _) in enumerate(postings):
            if existingDocID > docID:
                insertIndex = i
                break

        postings.insert(insertIndex, (docID, [position]))

import mimetypes

def application(environ, start_response):
    try:
        path = environ.get('PATH_INFO', '')
        query_params = parse_qs(environ.get('QUERY_STRING', ''))
        status = '200 OK'
        headers = [('Content-Type', 'application/json'), ('Access-Control-Allow-Origin', '*')]

        # Serve static files from /public
        if path == '/' or path == '/index.html':
            file_path = os.path.join('public', 'index.html')
        else:
            file_path = os.path.join('public', path.lstrip('/'))

        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or 'text/plain'
            headers = [('Content-Type', mime_type)]
            start_response('200 OK', headers)
            with open(file_path, 'rb') as f:
                return [f.read()]

        # Search endpoint
        if path == '/search':
            query = query_params.get('q', [''])[0]
            method = query_params.get('method', ['term'])[0]

            if not query:
                status = '400 Bad Request'
                response = {'error': 'Missing query parameter "q"'}
            else:
                start_time = time.time()
                if method == 'term':
                    results = indexer.queryWithTerm(query)
                    response = {'method': 'term', 'query': query, 'results': results}
                elif method == 'tfidf':
                    results, _ = indexer.queryWithTfIdf(query)
                    results = list(map(lambda x: x[0], results))
                    response = {'method': 'tfidf', 'query': query, 'results': results}
                else:
                    status = '400 Bad Request'
                    response = {'error': 'Invalid method. Use "term" or "tfidf".'}
                response['elapsed_time'] = round(time.time() - start_time, 4)

        # Document endpoint
        elif path == '/document':
            doc_id = query_params.get('id', [None])[0]
            if doc_id is None:
                status = '400 Bad Request'
                response = {'error': 'Missing document ID parameter "id"'}
            else:
                try:
                    doc_id = int(doc_id)
                except ValueError:
                    status = '400 Bad Request'
                    response = {'error': 'Invalid document ID format. Must be an integer.'}
                else:
                    collection_path = indexer.collectionPath
                    if not os.path.exists(collection_path):
                        status = '500 Internal Server Error'
                        response = {'error': f'Collection file not found at {collection_path}'}
                    else:
                        # Simplified and non-blocking XML parsing
                        with open(collection_path, 'r') as f:
                            content = f.read()

                        # Use regex to extract document by ID
                        import re
                        pattern = re.compile(rf"<docno>\s*{doc_id}\s*</docno>.*?<headline>(.*?)</headline>.*?<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
                        match = pattern.search(content)
                        if match:
                            headline = match.group(1).strip()
                            text = match.group(2).strip()
                            response = {'doc_id': doc_id, 'headline': headline, 'content': text}
                        else:
                            status = '404 Not Found'
                            response = {'error': f'Document with ID {doc_id} not found'}

        else:
            status = '404 Not Found'
            response = {'error': 'Invalid endpoint'}

    except Exception as e:
        status = '500 Internal Server Error'
        response = {'error': str(e)}

    start_response(status, headers)
    return [json.dumps(response).encode('utf-8')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-path', type=str, default='collections/trec.sample.xml', help='Path to search in')
    parser.add_argument('--index-path', type=str, default='index.txt', help='Path to index file')
    parser.add_argument('--stopword-path', type=str, default='stop_words.txt', help='Path to stopword file')
    parser.add_argument('--mode', default='server', choices=['server', 'console'], help='Mode to run the search engine')
    parser.add_argument('--search-method', default='term', choices=['term', 'tfidf'], help='Search method to use')
    args = parser.parse_args()

    if args.mode == 'server':
        indexer = Indexer('collections/trec.sample.xml', 'index.txt', 'stop_words.txt')
        if not os.path.isfile('index.txt'):
            print(f"Index file '{args.index_path}' does not exist. Building index...")
            startTime = time.time()
            indexer.buildIndex()
            endTime = time.time()
            print(f"Generate the index file for {endTime - startTime} seconds.")
        else:
            print(f"Index file '{args.index_path}' already exists. Skipping indexing process.")
            startTime = time.time()
            indexer.loadIndex()
            endTime = time.time()
            print(f"Load the index file for {endTime - startTime} seconds.")
        with make_server('0.0.0.0', 5050, application) as httpd:
            print("Serving integrated web UI and API on port 5050...")
            print("Visit http://localhost:5050 in your browser.")
            httpd.serve_forever()
    elif args.mode == 'console':
        if not args.collection_path.endswith('.xml'):
            raise ValueError("The path must point to a XML file.")

        collectionFile = open(args.collection_path, 'r')
        indexer = Indexer(args.collection_path, args.index_path, args.stopword_path)

        if not os.path.isfile(args.index_path):
            print(f"Index file '{args.index_path}' does not exist. Building index...")
            startTime = time.time()
            indexer.buildIndex()
            endTime = time.time()
            print(f"Generate the index file for {endTime - startTime} seconds.")
        else:
            print(f"Index file '{args.index_path}' already exists. Skipping indexing process.")
            startTime = time.time()
            indexer.loadIndex()
            endTime = time.time()
            print(f"Load the index file for {endTime - startTime} seconds.")

        print(f"Index loaded with {len(indexer.index)} terms.")
        if args.search_method == 'term':
            print('Enter your search query with term search (or type "EXIT" to quit):')

            while True:
                query = input("> ")
                if query == 'EXIT':
                    print('Good Bye!')
                    exit()

                docIDs = indexer.queryWithTerm(query)
                if docIDs != -1:
                    print(docIDs)
                    print(f'Total {len(docIDs)} documents found.')
        elif args.search_method == 'tfidf':
            print('Enter your search query with TFIDF search (or type "EXIT" to quit):')

            while True:
                query = input("> ")
                if query == 'EXIT':
                    print('Good Bye!')
                    exit()

                results, suggestedTerms = indexer.queryWithTfIdf(query)
                print(f'Top Documents: {results[:5]}')
                print(f'Suggested Terms: {suggestedTerms}')
                print(f'Total {len(results)} documents found.')
