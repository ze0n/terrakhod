curl -L https://files.slack.com/files-pri/T2CKKMS1K-FCYMCB533/download/good.zip?pub_secret=1c4f04ba90 --output %1%/2018-IRONCAR-SHARED-1-250x150-JPEGS.zip
mkdir %1/2018-IRONCAR-SHARED-1-250x150-JPEGS
unzip %1/2018-IRONCAR-SHARED-1-250x150-JPEGS.zip -d %1/2018-IRONCAR-SHARED-1-250x150-JPEGS

curl -L https://files.slack.com/files-pri/T2CKKMS1K-FCYMA92DP/download/records.zip?pub_secret=25e23f09c8 --output %1%/2018-IRONCAR-SHARED-2-250x150-JPEGS.zip
mkdir %1/2018-IRONCAR-SHARED-2-250x150-JPEGS
unzip %1/2018-IRONCAR-SHARED-2-250x150-JPEGS.zip -d %1/2018-IRONCAR-SHARED-2-250x150-JPEGS

curl -L https://s3.amazonaws.com/axionautdataset/Datasets+2.zip --output %1%/2018-AXIONAUT.zip
mkdir %1/2018-AXIONAUT.zip
unzip %1/2018-AXIONAUT.zip -d %1/2018-AXIONAUT

echo "Done" > %1/downloaded.txt