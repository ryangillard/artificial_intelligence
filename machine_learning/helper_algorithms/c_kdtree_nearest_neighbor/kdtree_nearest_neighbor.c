#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// A structure to represent a node of k-d tree
struct Node
{
    unsigned int index;
    unsigned int depth;
    unsigned int axis; // splitting axis
    double value; // splitting value
    double *point; // to store k dimensional point
    struct Node *parent, *left, *right;
} *root = NULL, *temp = NULL, *parent = NULL;

// A structure to represent k dimensional points
struct points
{
    unsigned int index;
    double *point; // to store k dimensional point
};

// This function creates a k-d tree node
struct Node* createNewKDTreeNode(struct Node *parent, struct Node *root, unsigned int number_of_dimensions, struct points *source_points, unsigned int median_index, unsigned int depth, unsigned int axis);

// This function inserts a new node and returns root of modified k-d tree. The parameter depth is used to decide axis of comparison.
struct Node *insertNewKDTreeNode(struct Node *parent, struct Node *root, unsigned int number_of_dimensions, struct points *source_points, unsigned int depth, unsigned int start_index, unsigned int end_index);

// This function searches for nearest node in the k-d tree to target point. The parameter depth is used to determine current axis.
void nearestNeighborSearch(struct Node* root, unsigned int number_of_dimensions, struct points *target_point, unsigned int depth, double *shortest_distance, struct points *closest_source_point_to_target_point);

/* recursive function to perform inOrder traversal of tree */
void inOrder(struct Node *root, unsigned int number_of_dimensions);

/* To find the preOrder traversal */
void preOrder(struct Node *root, unsigned int number_of_dimensions);

/* To find the postOrder traversal */
void postOrder(struct Node *root, unsigned int number_of_dimensions);

// This function deletes the k-d tree
void deleteKDTree(struct Node *root);

// This function applies quicksort to doubles and keeps track of the change in indexing
void quicksortDoubles (unsigned int n, double *a, unsigned int *index);

// Driver program to test above functions
int main (int argc, char *argv[])
{
    unsigned int i, j;
    int systemreturn = 0;

    // Get the number of number_of_dimensions
    unsigned int number_of_dimensions = 0;
    
    FILE* infile_number_of_dimensions = fopen("inputs/number_of_dimensions.txt", "r");
    if (infile_number_of_dimensions == NULL)
    {
        printf("Failed to open file inputs/number_of_dimensions.txt\n");
    }
    else
    {
        systemreturn = fscanf(infile_number_of_dimensions, "%u", &number_of_dimensions);
        if (systemreturn == -1)
        {
            printf("Failed reading file inputs/number_of_dimensions.txt\n");
        }
        fclose(infile_number_of_dimensions);
    }
    
    // Get the number of source points
    unsigned int number_of_source_points = 0;
    
    FILE* infile_number_of_source_points = fopen("inputs/number_of_source_points.txt", "r");
    if (infile_number_of_source_points == NULL)
    {
        printf("Failed to open file inputs/number_of_source_points.txt\n");
    }
    else
    {
        systemreturn = fscanf(infile_number_of_source_points, "%u", &number_of_source_points);
        if (systemreturn == -1)
        {
            printf("Failed reading file inputs/number_of_source_points.txt\n");
        }
        fclose(infile_number_of_source_points);
    }
    
    // Get the number of target points
    unsigned int number_of_target_points = 0;
    
    FILE* infile_number_of_target_points = fopen("inputs/number_of_target_points.txt", "r");
    if (infile_number_of_target_points == NULL)
    {
        printf("Failed to open file inputs/number_of_target_points.txt\n");
    }
    else
    {
        systemreturn = fscanf(infile_number_of_target_points, "%u", &number_of_target_points);
        if (systemreturn == -1)
        {
            printf("Failed reading file inputs/number_of_target_points.txt\n");
        }
        fclose(infile_number_of_target_points);
    }
    
    // Get the source points
    struct points *source_points;
    source_points = malloc(sizeof(struct points) * number_of_source_points);
    
    FILE* infile_source_points = fopen("inputs/source_points.txt", "r");
    if (infile_source_points == NULL)
    {
        printf("Failed to open file inputs/source_points.txt\n");
    }
    else
    {
        for (i = 0; i < number_of_source_points; i++)
        {
            source_points[i].index = i;
            source_points[i].point = malloc(sizeof(double) * number_of_dimensions);
            for (j = 0; j < number_of_dimensions; j++)
            {
                systemreturn = fscanf(infile_source_points, "%lf\t", &source_points[i].point[j]);
                if (systemreturn == -1)
                {
                    printf("Failed reading file inputs/source_points.txt\n");
                }
                printf("i = %u, j = %u, source_points[i].index = %u, source_points[i].point[j] = %lf\n", i, j, source_points[i].index, source_points[i].point[j]);
            } // end of j loop
        } // end of i loop
        fclose(infile_source_points);
    }
    
    // Get the target points
    struct points *target_points;
    target_points = malloc(sizeof(struct points) * number_of_target_points);
    
    FILE* infile_target_points = fopen("inputs/target_points.txt", "r");
    if (infile_target_points == NULL)
    {
        printf("Failed to open file inputs/target_points.txt\n");
    }
    else
    {
        for (i = 0; i < number_of_target_points; i++)
        {
            target_points[i].index = i;
            target_points[i].point = malloc(sizeof(double) * number_of_dimensions);
            for (j = 0; j < number_of_dimensions; j++)
            {
                systemreturn = fscanf(infile_target_points, "%lf\t", &target_points[i].point[j]);
                if (systemreturn == -1)
                {
                    printf("Failed reading file inputs/target_points.txt\n");
                }
                printf("i = %u, j = %u, target_points[i].index = %u, target_points[i].point[j] = %lf\n", i, j, target_points[i].index, target_points[i].point[j]);
            } // end of j loop
        } // end of i loop
        fclose(infile_target_points);
    }
    
    printf("main: number_of_dimensions = %u, number_of_source_points = %u, number_of_target_points = %u\n", number_of_dimensions, number_of_source_points, number_of_target_points);

    // Create k-d tree from source points
    root = insertNewKDTreeNode(parent, root, number_of_dimensions, source_points, 0, 0, number_of_source_points);
    printf("\n");
    
    // Create array to keep track of the closest source point to each target point
    struct points *closest_source_point_to_target_point;
    closest_source_point_to_target_point = malloc(sizeof(struct points) * number_of_target_points);
    for (i = 0; i < number_of_target_points; i++)
    {
        closest_source_point_to_target_point[i].index = 0;
        closest_source_point_to_target_point[i].point = malloc(sizeof(double) * number_of_dimensions);
        for (j = 0; j < number_of_dimensions; j++)
        {
            closest_source_point_to_target_point[i].point[j] = DBL_MAX;
        } // end of j loop
    } // end of i loop
    
    // Find nearest neighbor of target point within source point k-d tree
    double shortest_distance;
    for (i = 0; i < number_of_target_points; i++)
    {
        shortest_distance = DBL_MAX;
        
        nearestNeighborSearch(root, number_of_dimensions, &target_points[i], 0, &shortest_distance, &closest_source_point_to_target_point[i]);
        
        printf("shortest_distance = %lf, closest_source_point_to_target_point[i].index = %u", shortest_distance, closest_source_point_to_target_point[i].index);
        for (j = 0; j < number_of_dimensions; j++)
        {
            printf(", closest_source_point_to_target_point[i].point[j] = %lf", closest_source_point_to_target_point[i].point[j]);
        } // end of j loop
        printf("\n");
    } // end of i loop

    // Print trees
    printf("\nmain: root:\n");
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("%lf\t", root->point[i]);
    } // end of i loop

    // inOrder tree traversal
    printf("\nmain: inOrder:\n");
    printf("index\tdepth\taxis\tvalue");
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tpoint[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tparent[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tleft[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tright[%u]", i);
    } // end of i loop
    printf("\n");
    inOrder(root, number_of_dimensions);

    // preOrder tree traversal
    printf("\nmain: preOrder:\n");
    printf("index\tdepth\taxis\tvalue");
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tpoint[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tparent[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tleft[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tright[%u]", i);
    } // end of i loop
    printf("\n");
    preOrder(root, number_of_dimensions);

    // postOrder tree traversal
    printf("\nmain: postOrder:\n");
    printf("index\tdepth\taxis\tvalue");
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tpoint[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tparent[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tleft[%u]", i);
    } // end of i loop
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\tright[%u]", i);
    } // end of i loop
    printf("\n");
    postOrder(root, number_of_dimensions);

    // Free memory
    printf("main: Deleting tree\n");
    deleteKDTree(root);
    
    printf("main: Freeing points array memory\n");
    for (i = 0; i < number_of_target_points; i++)
    {
        free(closest_source_point_to_target_point[i].point);
        
        free(target_points[i].point);
    }
    free(closest_source_point_to_target_point);
    
    free(target_points);
    
    for (i = 0; i < number_of_source_points; i++)
    {
        free(source_points[i].point);
    }
    free(source_points);
    
    printf("main: Program done\n");

    return 0;
} // end of main

// This function creates a k-d tree node
struct Node* createNewKDTreeNode(struct Node *parent, struct Node *root, unsigned int number_of_dimensions, struct points *source_points, unsigned int median_index, unsigned int depth, unsigned int axis)
{
    unsigned int i;

    temp = malloc(sizeof(struct Node) * 1);
    temp->point = malloc(sizeof(double) * number_of_dimensions);

    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("createNewKDTreeNode: i = %u, median_index = %u, source_points[median_index].point[i] = %lf\n", i, median_index, source_points[median_index].point[i]);
    }

    temp->index = source_points[median_index].index;

    for (i = 0; i < number_of_dimensions; i++)
    {
        temp->point[i] = source_points[median_index].point[i];
    }

    temp->depth = depth;
    temp->axis = axis;
    temp->value = temp->point[axis];

    temp->parent = parent;
    temp->left = NULL;
    temp->right = NULL;

    return temp;
} // end of createNewKDTreeNode function

// This function inserts a new node and returns root of modified k-d tree. The parameter depth is used to decide axis of comparison.
struct Node *insertNewKDTreeNode(struct Node *parent, struct Node *root, unsigned int number_of_dimensions, struct points *source_points, unsigned int depth, unsigned int start_index, unsigned int end_index)
{
    unsigned int i, j;

    // Select axis based on depth so that axis cycles through all valid values
    unsigned int axis = depth % number_of_dimensions;

    printf("insertNewKDTreeNode: Entering function, depth = %u, number_of_dimensions = %u, axis = %u, start_index = %u, end_index = %u\n", depth, number_of_dimensions, axis, start_index, end_index);

    // Sort point list and choose median as pivot element
    double *axis_values;
    axis_values = malloc(sizeof(double) * end_index);
    
    unsigned int *axis_indices;
    axis_indices = malloc(sizeof(unsigned int) * end_index);

    for (i = 0; i < end_index; i++)
    {
        axis_values[i] = source_points[i].point[axis];
        axis_indices[i] = i;
        
        printf("insertNewKDTreeNode: i = %u, axis_values[i] = %lf\n", i, axis_values[i]);
        printf("insertNewKDTreeNode: i = %u, axis_indices[i] = %u\n", i, axis_indices[i]);
    } // end of i loop

    // Quicksort to find median
    quicksortDoubles(end_index, axis_values, axis_indices);

    unsigned int median_index = end_index / 2;

    struct points *sorted_points;
    sorted_points = malloc(sizeof(struct points) * end_index);
    for (i = 0; i < end_index; i++)
    {
        sorted_points[i].index = source_points[axis_indices[i]].index;
        printf("%u\t%u", i, sorted_points[i].index);
        sorted_points[i].point = malloc(sizeof(double) * number_of_dimensions);
        
        for (j = 0; j < number_of_dimensions; j++)
        {
            sorted_points[i].point[j] = source_points[axis_indices[i]].point[j];
            printf("\t%lf", sorted_points[i].point[j]);
        } // end of j loop
        printf("\n");
    } // end of i loop

    // Create node and construct subtrees
    if (root == NULL)
    {
        printf("insertNewKDTreeNode: Creating node!\n");
        root = createNewKDTreeNode(parent, root, number_of_dimensions, sorted_points, median_index, depth, axis);
    }

    if (median_index > 0)
    {
        root->left  = insertNewKDTreeNode(root, root->left, number_of_dimensions, &sorted_points[0], depth + 1, 0, median_index);
    }

    if (end_index - (median_index + 1) > 0)
    {
        root->right = insertNewKDTreeNode(root, root->right, number_of_dimensions, &sorted_points[median_index + 1], depth + 1, median_index + 1, end_index - (median_index + 1));
    }
    
    // Free memory
    for (i = 0; i < end_index; i++)
    {
        free(sorted_points[i].point);
    }
    free(sorted_points);
    
    free(axis_indices);
    free(axis_values);
    printf("insertNewKDTreeNode: Exiting function, depth = %u, number_of_dimensions = %u, axis = %u, start_index = %u, end_index = %u\n", depth, number_of_dimensions, axis, start_index, end_index);

    return root;
} // end of insertNewKDTreeNode function

// This function searches for nearest node in the k-d tree to target point. The parameter depth is used to determine current axis.
void nearestNeighborSearch(struct Node* root, unsigned int number_of_dimensions, struct points *target_point, unsigned int depth, double *shortest_distance, struct points *closest_source_point_to_target_point)
{
    // Base cases
    if (root == NULL)
    {
        return;
    }

    unsigned int i;

    // Select axis based on depth so that axis cycles through all valid values
    unsigned int axis = depth % number_of_dimensions;

    printf("nearestNeighborSearch: Entering function, depth = %u, axis = %u, (*shortest_distance) = %lf, closest_source_point_to_target_point->index = %u\n", depth, axis, (*shortest_distance), closest_source_point_to_target_point->index); // going into scope down the tree

    if (root->right == NULL && root->left == NULL)
    {
        if ((*shortest_distance) == DBL_MAX) // if we have NOT descended to a leaf node yet
        {
            printf("nearestNeighborSearch: This is a leaf node! depth = %u, axis = %u\n", root->depth, root->axis);
            for (i = 0; i < number_of_dimensions; i++)
            {
                printf("%lf\t", root->point[i]);
            } // end of i loop
            printf("\n");

            if ((*shortest_distance) == DBL_MAX)
            {
                double difference;
                (*shortest_distance) = 0;
                closest_source_point_to_target_point->index = root->index;
                for (i = 0; i < number_of_dimensions; i++)
                {
                    difference = (root->point[i] - target_point->point[i]);
                    (*shortest_distance) += (difference * difference);
                    closest_source_point_to_target_point->point[i] = root->point[i];
                } // end of i loop
            }
        }
        else
        {
            printf("nearestNeighborSearch: This is a leaf node! Descended to leaf node already! depth = %u, axis = %u\n", root->depth, root->axis);
            for (i = 0; i < number_of_dimensions; i++)
            {
                printf("%lf\t", root->point[i]);
            } // end of i loop
            printf("\n");

            double difference, distance;
            distance = 0;
            for (i = 0; i < number_of_dimensions; i++)
            {
                difference = (root->point[i] - target_point->point[i]);
                distance += (difference * difference);
            } // end of i loop

            printf("nearestNeighborSearch: distance = %lf\n", distance);

            if (distance < (*shortest_distance))
            {
                (*shortest_distance) = distance;
                closest_source_point_to_target_point->index = root->index;
                for (i = 0; i < number_of_dimensions; i++)
                {
                    closest_source_point_to_target_point->point[i] = root->point[i];
                } // end of i loop
            }
        }
    }
    else
    {
        printf("nearestNeighborSearch: This is NOT a leaf node! depth = %u, axis = %u\n", root->depth, root->axis);
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("%lf\t", root->point[i]);
        } // end of i loop
        printf("\n");

        // Compare point with root with respect to axis
        if (target_point->point[axis] < root->point[axis])
        {
            if (root->left != NULL)
            {
                nearestNeighborSearch(root->left, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
            }
            else
            {
                nearestNeighborSearch(root->right, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
            }
        }
        else
        {
            if (root->right != NULL)
            {
                nearestNeighborSearch(root->right, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
            }
            else
            {
                nearestNeighborSearch(root->left, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
            }
        }

        if ((*shortest_distance) != DBL_MAX) // if we have descended to a leaf node already
        {
            printf("nearestNeighborSearch: This is NOT a leaf node! Descended to leaf node already! depth = %u, axis = %u\n", root->depth, root->axis);
            for (i = 0; i < number_of_dimensions; i++)
            {
                printf("%lf\t", root->point[i]);
            } // end of i loop
            printf("\n");

            double difference, distance;
            distance = 0;
            for (i = 0; i < number_of_dimensions; i++)
            {
                difference = (root->point[i] - target_point->point[i]);
                distance += (difference * difference);
            } // end of i loop

            printf("nearestNeighborSearch: distance = %lf\n", distance);

            if (distance < (*shortest_distance))
            {
                (*shortest_distance) = distance;
                closest_source_point_to_target_point->index = root->index;
                for (i = 0; i < number_of_dimensions; i++)
                {
                    closest_source_point_to_target_point->point[i] = root->point[i];
                } // end of i loop
            }
            else
            {
                difference = root->point[axis];
                distance = difference * difference;
                printf("nearestNeighborSearch: Hypersphere distance = %lf\n", distance);
                if ((*shortest_distance) > distance) // if hypersphere crosses the plane
                {
                    printf("nearestNeighborSearch: Hypersphere crosses the plane!\n");
                    // Compare point with root with respect to axis
                    if (target_point->point[axis] < root->point[axis])
                    {
                        printf("nearestNeighborSearch: Going back down to the right!\n");
                        nearestNeighborSearch(root->right, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
                    }
                    else
                    {
                        printf("nearestNeighborSearch: Going back down to the left!\n");
                        nearestNeighborSearch(root->left, number_of_dimensions, target_point, depth + 1, shortest_distance, closest_source_point_to_target_point);
                    }
                }
            }
        }
    }

    printf("nearestNeighborSearch: Exiting function, depth = %u, axis = %u, (*shortest_distance) = %lf, closest_source_point_to_target_point->index = %u\n", depth, axis, (*shortest_distance), closest_source_point_to_target_point->index); // coming out of scope back up the tree

} // end of nearestNeighborSearch function


// This function recursively performs in-order traversal of tree
void inOrder(struct Node *root, unsigned int number_of_dimensions)
{
    if (root == NULL)
    {
        printf("inOrder: No elements in a tree to display");
        return;
    }

    if (root->left != NULL)
    {
        inOrder(root->left, number_of_dimensions);
    }

    printf("%u\t%u\t%u\t%lf", root->index, root->depth, root->axis, root->value);
    unsigned int i;
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\t%lf", root->point[i]);
    }

    if (root->parent != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->parent->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->left != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->left->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->right != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->right->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }
    printf("\n");

    if (root->right != NULL)
    {
        inOrder(root->right, number_of_dimensions);
    }
} // end of inOrder function

// This function recursively performs pre-order traversal of tree
void preOrder(struct Node *root, unsigned int number_of_dimensions)
{
    if (root == NULL)
    {
        printf("preOrder: No elements in a tree to display");
        return;
    }

    printf("%u\t%u\t%u\t%lf", root->index, root->depth, root->axis, root->value);
    unsigned int i;
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\t%lf", root->point[i]);
    }

    if (root->parent != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->parent->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->left != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->left->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->right != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->right->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }
    printf("\n");

    if (root->left != NULL)
    {
        preOrder(root->left, number_of_dimensions);
    }

    if (root->right != NULL)
    {
        preOrder(root->right, number_of_dimensions);
    }
} // end of preOrder function

// This function recursively performs post-order traversal of tree
void postOrder(struct Node *root, unsigned int number_of_dimensions)
{
    if (root == NULL)
    {
        printf("postOrder: No elements in a tree to display ");
        return;
    }

    if (root->left != NULL)
    {
        postOrder(root->left, number_of_dimensions);
    }

    if (root->right != NULL)
    {
        postOrder(root->right, number_of_dimensions);
    }

    printf("%u\t%u\t%u\t%lf", root->index, root->depth, root->axis, root->value);
    unsigned int i;
    for (i = 0; i < number_of_dimensions; i++)
    {
        printf("\t%lf", root->point[i]);
    }

    if (root->parent != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->parent->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->left != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->left->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }

    if (root->right != NULL)
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\t%lf", root->right->point[i]);
        }
    }
    else
    {
        for (i = 0; i < number_of_dimensions; i++)
        {
            printf("\tNULL");
        }
    }
    printf("\n");
} // end of postOrder function

// This function deletes the K D tree
void deleteKDTree(struct Node *root)
{
    if (root == NULL)
    {
        printf("deleteKDTree: No elements in a tree to display ");
        return;
    }

    if (root->left != NULL)
    {
        deleteKDTree(root->left);
    }

    if (root->right != NULL)
    {
        deleteKDTree(root->right);
    }

    free(root->point);
    free(root);
} // end of postOrder function

// This function applies quicksort to doubles and keeps track of the change in indexing
void quicksortDoubles (unsigned int n, double *a, unsigned int *index)
{
    unsigned int i, j;
    double p, t;

    if (n < 2)
    {
        return;
    }

    // Choose pivot as midpoint of a
    p = a[n / 2];

    for (i = 0, j = n - 1;; i++, j--)
    {
        while (a[i] < p)
        {
            i++;
        }

        while (p < a[j])
        {
            j--;
        }

        if (i >= j)
        {
            break;
        }

        // Swap
        t = a[i];
        a[i] = a[j];
        a[j] = t;

        t = index[i];
        index[i] = index[j];
        index[j] = t;
    } // end of i, j loop

    quicksortDoubles (i, a, index);
    quicksortDoubles (n - i, a + i, index + i);
} // end of quicksortDoubles function