#!/bin/bash
#
# Copyright 2022 Dumitrel Loghin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# kill and remove docker containes
#

IMGNAME="hbdb"
PREFIX="hbdb"

idx=1
for id in `docker ps | grep $PREFIX | cut -d ' ' -f 1`; do
	echo "$PREFIX$idx"
	idx=$(($idx+1))
	docker kill $id
	docker rm $id
done
