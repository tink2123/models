DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# Extract the data.
echo "Extracting..."
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip

# Get valid 5k division
echo "Get valid 5k division..."
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

# Get label
echo "Get label..."
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
