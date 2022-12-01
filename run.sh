# array=(
#     60
#     300
#     300
#     300
#     300
#     300
#     300
#     300
#     900
# )
# array2=(
#     0
#     0
#     30
#     60
#     90
#     300
#     900
#     1800
#     0
# )

array=(
    300
)
array2=(
    0
)


for index in ${!array[*]}; do
    for i in {1..143}
        do
            echo $i "window ${array[$index]} distance ${array2[$index]}"
            python src/main.py ${array[$index]} ${array2[$index]}
        done 
done