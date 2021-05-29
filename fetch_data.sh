#!/bin/bash

kaggle datasets download -d johnowhitaker/opencitiesTilesMasked
unzip -q opencitiesTilesMasked.zip
rm opencitiesTilesMasked.zip
