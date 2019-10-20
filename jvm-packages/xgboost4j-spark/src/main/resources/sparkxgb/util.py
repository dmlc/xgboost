#
# Copyright (c) 2019 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pyspark.ml.util import JavaMLReadable, JavaMLReader


class XGBoostReadable(JavaMLReadable):
    """
    Mixin class that provides a read() method for XGBoostReader.
    """

    @classmethod
    def read(cls):
        """Returns an XGBoostReader instance for this class."""
        return XGBoostReader(cls)


class XGBoostReader(JavaMLReader):
    """
    A reader mixin class for XGBoost objects.
    """

    @classmethod
    def _java_loader_class(cls, clazz):
        if hasattr(clazz, '_java_class_name') and clazz._java_class_name is not None:
            return clazz._java_class_name
        else:
            return JavaMLReader._java_loader_class(clazz)
