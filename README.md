# Clone & enter repo
git clone https://github.com/ImaginaryBond7/review_extractor.git
cd review_extractor

# Create and activate virtual env
python -m venv venv
source venv/bin/activate    # for Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Add your OpenAI key
echo "OPENAI_API_KEY={YOUR_OPENAI_API_KEY}" > .env

# To run testcases
pytest -q

# For help
python3 -m cli.cli --help

# To extract delight attribute
python3 -m cli.cli extract \
  --input {INPUT_PATH=reviews.json} \
  --output {OUTPUT_PATH=reviews_with_attributes.json} \
  --csv-output {OUTPUT_PATH=ranked_attributes.csv}

# To evaluate accuracy
python3 -m cli.cli evaluate \ 
  --csv {INPUT_PATH=delight-evaluation.csv} \
  --report {OUTPUT_PATH=eval.json}