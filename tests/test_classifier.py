import unittest

from classifier import (
    DATA_PATH,
    classify_text,
    extract_document_text,
    load_training_examples,
    train_classifier,
)


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts, labels = load_training_examples(DATA_PATH)
        cls.model = train_classifier(texts, labels)

    def test_training_data_contains_four_document_types(self):
        _, labels = load_training_examples(DATA_PATH)
        self.assertEqual(set(labels), {"invoice", "resume", "contract", "letter"})

    def test_recognizes_clear_examples(self):
        examples = {
            "invoice": "Invoice total due is $80. Please pay by the due date.",
            "resume": "My work experience includes Python and software development skills.",
            "contract": "Both parties agree to the terms and conditions of this signed agreement.",
            "letter": "Dear Maria, I am writing to invite you to our school event.",
        }
        for expected_label, text in examples.items():
            with self.subTest(label=expected_label):
                guess, score, scores = classify_text(self.model, text)
                self.assertEqual(guess, expected_label)
                self.assertGreater(score, 0)
                self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "add some document text"):
            classify_text(self.model, "   ")

    def test_extracts_text_files(self):
        text = extract_document_text("note.txt", b"Hello from a tiny text file.")
        self.assertEqual(text, "Hello from a tiny text file.")

    def test_rejects_unknown_file_type(self):
        with self.assertRaisesRegex(ValueError, "choose a .txt, .pdf, or .docx"):
            extract_document_text("photo.png", b"not a document")


if __name__ == "__main__":
    unittest.main()
