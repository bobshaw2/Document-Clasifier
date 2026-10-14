import pandas as pd
import streamlit as st

from classifier import (
    DATA_PATH,
    classify_text,
    extract_document_text,
    load_training_examples,
    train_classifier,
)

st.set_page_config(
    page_title="Document sorter",
    page_icon=":material/description:",
    layout="centered",
)


@st.cache_resource
def get_model():
    texts, labels = load_training_examples(DATA_PATH)
    return train_classifier(texts, labels)


model = get_model()
labels = list(model.named_steps["classifier"].classes_)

with st.sidebar:
    st.header("Your mission")
    st.write("Give the computer a document. It will guess which kind it is.")
    st.caption("This practice app uses a small local model, not an AI service.")
    st.write("**It knows four kinds:**")
    for label in labels:
        st.write(f"- {label}")

st.title("Document sorter", icon=":material/description:")
st.write("Drop in a document, and our little text-sorting machine will make its best guess.")

with st.container(border=True):
    st.subheader("1. Give me a document")
    input_method = st.segmented_control(
        "How would you like to add it?",
        options=["Paste text", "Upload a file"],
        default="Paste text",
        key="input_method",
    )

    document_text = ""
    if input_method == "Upload a file":
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["txt", "pdf", "docx"],
            help="Text files, PDFs, and Word documents are supported.",
        )
        if uploaded_file is not None:
            try:
                document_text = extract_document_text(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                st.success(f"Read {uploaded_file.name}")
                st.text_area(
                    "Text found in your document",
                    value=document_text[:5000],
                    height=180,
                    disabled=True,
                )
                if len(document_text) > 5000:
                    st.caption("Showing the first 5,000 characters here; the whole document will be checked.")
            except ValueError as error:
                st.error(str(error))
    else:
        document_text = st.text_area(
            "Paste a few sentences from your document",
            placeholder="Example: Please pay the total amount of $125 by the due date...",
            height=180,
            max_chars=100_000,
        )

    classify_button = st.button(
        "Sort my document",
        type="primary",
        icon=":material/auto_awesome:",
        disabled=not document_text.strip(),
    )

if classify_button:
    try:
        prediction = classify_text(model, document_text)
        st.session_state["last_prediction"] = prediction
    except ValueError as error:
        st.error(str(error))

if "last_prediction" in st.session_state:
    predicted_label, confidence, scores = st.session_state["last_prediction"]
    st.space("small")
    with st.container(border=True):
        st.subheader("2. The computer's guess")
        st.metric("Best guess", predicted_label)
        st.progress(confidence, text=f"Model score: {confidence:.0%}")
        st.caption(
            "A score is how strongly this model leans toward a label. "
            "It is not a guarantee that the guess is correct."
        )

        score_table = pd.DataFrame(
            {
                "Document type": list(scores.keys()),
                "Model score": [f"{score:.1%}" for score in scores.values()],
            }
        ).sort_values("Document type")
        st.dataframe(score_table, hide_index=True)

with st.expander("How did the computer learn?", icon=":material/lightbulb:"):
    st.write(
        "We show it labeled examples, like flash cards. It notices which words "
        "often appear in each kind of document, then uses those clues to make a guess."
    )
    st.code(
        "examples, labels = load_training_examples()\n"
        "model = train_classifier(examples, labels)\n"
        "guess, score, all_scores = classify_text(model, my_document)",
        language="python",
    )
    st.caption(
        f"The practice deck has {len(load_training_examples()[0])} short examples. "
        "Real workplaces need many more examples and careful checking."
    )

with st.expander("What can I do if the guess is wrong?", icon=":material/help:"):
    st.write(
        "Open `sample_documents.csv`, add a new row with a document example and "
        "the correct label, save it, then restart the app. More varied examples "
        "help the computer learn better."
    )
    st.write("The allowed labels are **invoice**, **resume**, **contract**, and **letter**.")
