"""Pytest fixtures for Tuxedo tests."""

import pytest

# Sample TEI XML response from Grobid
SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title>Deep Learning for Natural Language Processing: A Survey</title>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic>
                        <author>
                            <persName>
                                <forename>John</forename>
                                <surname>Smith</surname>
                            </persName>
                            <affiliation>
                                <orgName>MIT</orgName>
                            </affiliation>
                        </author>
                        <author>
                            <persName>
                                <forename>Jane</forename>
                                <surname>Doe</surname>
                            </persName>
                            <affiliation>
                                <orgName>Stanford University</orgName>
                            </affiliation>
                        </author>
                        <idno type="DOI">10.1234/example.2024.001</idno>
                    </analytic>
                    <monogr>
                        <imprint>
                            <date when="2024-06-15"/>
                        </imprint>
                    </monogr>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
        <profileDesc>
            <abstract>
                <p>This paper presents a comprehensive survey of deep learning methods for natural language processing.</p>
                <p>We review transformer architectures, attention mechanisms, and pre-trained language models.</p>
            </abstract>
            <textClass>
                <keywords>
                    <term>deep learning</term>
                    <term>natural language processing</term>
                    <term>transformers</term>
                </keywords>
            </textClass>
        </profileDesc>
    </teiHeader>
    <text>
        <body>
            <div>
                <head>Introduction</head>
                <p>Natural language processing has been transformed by deep learning.</p>
                <p>This survey covers recent advances in the field.</p>
            </div>
            <div>
                <head>Methods</head>
                <p>We analyze transformer-based architectures.</p>
            </div>
            <div>
                <head>Conclusion</head>
                <p>Deep learning continues to advance NLP capabilities.</p>
            </div>
        </body>
    </text>
</TEI>
"""

# Minimal TEI XML with only required fields
MINIMAL_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title>A Short Paper</title>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic/>
                    <monogr>
                        <imprint/>
                    </monogr>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
        <profileDesc/>
    </teiHeader>
    <text>
        <body/>
    </text>
</TEI>
"""

# TEI XML with missing title (should fall back to filename)
TEI_XML_NO_TITLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title/>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic/>
                    <monogr>
                        <imprint/>
                    </monogr>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
        <profileDesc/>
    </teiHeader>
    <text>
        <body/>
    </text>
</TEI>
"""


@pytest.fixture
def sample_tei_xml():
    """Return sample TEI XML for testing."""
    return SAMPLE_TEI_XML


@pytest.fixture
def minimal_tei_xml():
    """Return minimal TEI XML for testing."""
    return MINIMAL_TEI_XML


@pytest.fixture
def tei_xml_no_title():
    """Return TEI XML without title for testing."""
    return TEI_XML_NO_TITLE


@pytest.fixture
def sample_pdf_content():
    """Return sample PDF content (just bytes for hashing)."""
    return b"%PDF-1.4 sample pdf content for testing"


@pytest.fixture
def tmp_pdf(tmp_path, sample_pdf_content):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_paper.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path
