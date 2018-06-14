# Python with pyspark

## Usage
~~~
spark-submit <script-name> <appName> <master> <dataset> <output-file>
~~~
## Example
~~~
spark-submit DecissionTree.py DecissionTree local[*] path/to/dataset.txt Times.txt
~~~