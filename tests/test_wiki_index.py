import tempfile
import unittest
from pathlib import Path

from llm_wiki.wiki import wiki_index


class WikiIndexTests(unittest.TestCase):
    def test_indexes_descriptions_and_refreshes_without_indexing_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wiki = root / "wiki"
            document = wiki / "论文"
            concepts = document / "concepts"
            concepts.mkdir(parents=True)
            (document / "entities").mkdir()
            methods = document / "methods" / "nested"
            methods.mkdir(parents=True)
            (wiki / "workflow-log").mkdir()
            (wiki / "workflow-log" / "verify-log.md").write_text("log")
            pages = {
                document / "source-summary-123456.md": "来源描述",
                concepts / "b.md": "第二个概念",
                concepts / "a.md": "第一个概念",
                methods / "method.md": "方法描述",
            }
            for path, description in pages.items():
                path.write_text(
                    f'---\ndescription: "{description}"\n---\n正文不进入索引\n',
                    encoding="utf-8",
                )
            self.assertTrue(wiki_index(root))
            self.assertEqual(
                (wiki / "index.md").read_text(), "- 论文: 来源描述\n"
            )
            self.assertEqual(
                (concepts / "index.md").read_text(),
                "- a.md: 第一个概念\n- b.md: 第二个概念\n",
            )
            self.assertEqual(
                (methods / "index.md").read_text(), "- method.md: 方法描述\n"
            )
            self.assertEqual((document / "entities" / "index.md").read_text(), "")
            self.assertFalse((wiki / "workflow-log" / "index.md").exists())
            (concepts / "a.md").unlink()
            self.assertTrue(wiki_index(root))
            self.assertEqual(
                (concepts / "index.md").read_text(), "- b.md: 第二个概念\n"
            )

    def test_invalid_description_does_not_replace_existing_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "wiki" / "source"
            document.mkdir(parents=True)
            (document / "source-summary-123456.md").write_text(
                '---\ndescription: ""\n---\n'
            )
            index = root / "wiki" / "index.md"
            index.write_text("existing index\n")
            self.assertFalse(wiki_index(root))
            self.assertEqual(index.read_text(), "existing index\n")


if __name__ == "__main__":
    unittest.main()
