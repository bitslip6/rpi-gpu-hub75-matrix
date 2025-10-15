#include <stddef.h.>

typedef struct string_t {
    uint32_t capacity; // total allocated size
    uint32_t length; // total allocated size
    uint32_t position;
    char *str; // the actual string data
}

typedef struct list_array_t {
    uint32_t position;
    size_t item_size;
    size_t length;
    size_t capacity;

    /**
     * get the pointer at index. returns NULL if out of bounds, else returns the pointer
     */
    (void *)(get(struct list_array_t *self, size_t index)) {
        if (index >= self->length) return NULL;
        return (void *)((char *)self->data + index * self->item_size);
    }

    /**
     * set the pointer at index. returns NULL if out of bounds, else returns the pointer
     * to the element in thge array. copies the data from item to the array.
     */
    (void *)(set_ptr(struct list_array_t *self, size_t index, void *item)) {
        if (index >= self->capacity) {
            return NULL;
        }
        return memcpy((char *)self->data + index * self->item_size, item, self->item_size);
    }

    /**
     * set the value at index. returns -1 if out of bounds, 0 if ok
     */
    (int)(set_value(struct list_array_t *self, size_t index, void value)) {
        if (index >= self->capacity) {
            return -1;
        }
        self->data[index] = value;
        return 0;
    }

    /**
     * create a new list_array_t with given item size and initial capacity
     */
    (list_array_t *)(create(size_t item_size, size_t initial_capacity)) {
        if (item_size == 0 || initial_capacity == 0) {
            return NULL;
        }
        size_t item_alloc = item_size * initial_capacity;
        list_array_t *arr = (list_array_t *)malloc(item_alloc + sizeof(list_array_t));
        if (!arr) {
            return NULL;
        }
        arr->item_size = item_size;
        arr->length = 0;
        arr->index = 0;
        arr->capacity = initial_capacity;
        return arr;
    }


    void *data;
};

