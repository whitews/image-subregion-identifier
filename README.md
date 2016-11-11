# image-subregion-identifier

Image Sub-region Identifier attempts to classify user specified regions using 
pre-trained data. Classification is based on Ridge regression from scikit-learn 
on image features extracted using [WND-CHARM](https://github.com/wnd-charm/wnd-charm).

## Usage

The application requires a pre-trained `data` directory with the following structure:

`data/[species]/[development]/[magnification]/[probes]/[class_name]/`

where:

* **species** is either `mouse` or `human`
* **development** is a valid development stage (e.g. `E16.5`)
* **magnification** is a valid magnification factor (e.g. `20X`)
* **probes** is an alphabetical list of probe strings separated by underscores (e.g. ``acta2_sftpc_sox9``)
* **class_name** is any text string for a given structure (e.g. `bronchiole`)

WND-CHARM signature files (.sig) must be placed inside each ``class_name`` sub-directory 
representing the training data. Finally, there must also be a metadata table in CSV format within the `data` 
directory with the name `metadata_table_all.csv`.
