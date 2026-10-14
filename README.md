# Document clasifier

A small learning project that sorts a document into one of four groups:
**invoice**, **resume**, **contract**, or **letter**.

## Think of it like sorting toys

Imagine four toy boxes. Each box has a name on it. We show the computer
examples that are already in the right boxes. The computer notices clues:
"amount due" often belongs in the invoice box, while "work experience" often
belongs in the resume box. Then it uses those clues to sort a new page.

This is a tiny practice model, not a judge. It can guess incorrectly, especially
when a document is unusual or when the example deck is small.

## Run it on Windows

Open PowerShell in this folder and type these one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Your browser should open the app. If PowerShell says that scripts are disabled,
you can run the app without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Learn it one bite at a time

1. **Find the examples.** Open `sample_documents.csv`. Each row has some text
   and its correct label, separated into columns.
2. **Read the words.** In `classifier.py`, `load_training_examples()` reads
   the CSV so Python gets a list of document text and a list of answers.
3. **Teach the computer.** `train_classifier()` builds a model. TF-IDF gives
   useful words scores, and logistic regression learns how those word clues
   relate to the labels.
4. **Ask it to sort.** `classify_text()` gives the model a new document and
   returns its best guess and the scores for all four labels.
5. **Show the result.** `streamlit_app.py` makes the buttons and result screen.
   Streamlit reruns the file when you click a button.

The main learning code is:

```python
texts, labels = load_training_examples()
model = train_classifier(texts, labels)
guess, score, all_scores = classify_text(model, "Please pay $25 by Friday.")
print(guess, score)
```

Try changing the sample sentence. Try adding a new, correctly labeled example
to `sample_documents.csv`, then restart the app.

## Add better examples

Every row needs two things: the document's words and its correct label. Add a
row like this (keeping the quotation marks around the text):

```csv
"Total payment due is $42 by Friday. Thank you for your purchase.",invoice
```

Add examples to the matching label, and keep the labels spelled exactly as
`invoice`, `resume`, `contract`, or `letter`. A real classifier needs many
different examples for each label and should be tested with documents it has
not seen before.

## Files

- `streamlit_app.py` — the user interface.
- `classifier.py` — the learning and document-reading code.
- `sample_documents.csv` — the tiny set of labeled teaching examples.
- `tests/test_classifier.py` — simple checks that the learner works.

Supported uploads: `.txt`, `.pdf`, and `.docx`. The app extracts text from
these files; scanned pictures inside PDFs need OCR, which this beginner project
does not include.

## Run the checks

```powershell
python -m unittest discover -s tests
```
