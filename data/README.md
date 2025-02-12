# How to downlaod data using the script

To download data run the following bash script:
```
bash get_data.sh <dataset name>[-record]
```
where `<dataset name>` could be:
- macho
- ogle
- atlas
- alcock

e.g.,
```
bash get_data.sh alcock
```
Notice you can add the `-record` tag at the end of the `<dataset name>`
Thus, in the previous example,
```
bash get_data.sh alcock-record
```
