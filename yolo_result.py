#!/usr/bin/env python3

"""
YoloResult class
"""


class YoloResult:
    """
    YoloResult
    """

    def __init__(self, class_index, obj_name, score, boundingbox):
        """
        Initialization method
        """

        self.class_index = class_index
        self.obj_name = obj_name.decode()
        self.score = score
        self.x_min = boundingbox[0] - boundingbox[2]/2 - 1
        self.y_min = boundingbox[1] - boundingbox[3]/2 - 1
        self.width = boundingbox[2]
        self.height = boundingbox[3]

    def get_detect_result(self):
        """
        getting yolo results
        return dict
        """

        resultdict = {
                      'class_index': self.class_index,
                      'obj_name': self.obj_name,
                      'score': self.score,
                      'bounding_box': {
                          'x_min': self.x_min,
                          'y_min': self.y_min,
                          'width': self.width,
                          'height': self.height}
                     }
        return resultdict

    def show(self):
        """
        show result
        """

        print('class_index : %d' % self.class_index)
        print('obj_name    : %s' % self.obj_name)
        print('score       : %.3f' % self.score)
        print('bbox.x_center  : %.2f' % self.x_min)
        print('bbox.y_center  : %.2f' % self.y_min)
        print('bbox.width  : %.2f' % self.width)
        print('bbox.height : %.2f' % self.height)


def main():
    """
    main
    """

    yolo_result = YoloResult(1, 'cat', 0.6,
                             [640*0.5, 360*0.5, 640*0.25, 340*0.25])
    yolo_result.show()


if __name__ == "__main__":
    main()
