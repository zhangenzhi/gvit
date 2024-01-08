import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
    
    def draw(self, ax, c='grey', lw=1, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), width=self.x2-self.x1, height=self.y2-self.y1, linewidth=lw, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
    
    
class QuadTree:
    def __init__(self, domain, bbox=None, depth=0, max_value=80, max_depth=6) -> None:
        self.domain = domain
        self.bbox = bbox
        self.depth = depth
        self.max_value = max_value
        self.max_depth = max_depth
        self.divided = False
        self.childs = None
        
        self._build_tree()
        
    def get_rect(self):
        return self.bbox.get_coord()
    
    def serialize(self, img):
        seq_patch = []
        if self.divided:
            for c_n in self.childs:
                seq_patch += c_n.serialize(img)
        else:
            seq_patch.append(self.bbox.get_area(img))
        return seq_patch
    
    def count_patches(self, patch_info):
        total_patches = 0
        if self.divided:
            for c in self.childs:
                total_patches +=  c.count_patches(patch_info)
        else:
            patch_size = self.bbox.get_size()
            key = "{}*{}".format(patch_size[0],patch_size[1])
            if key not in patch_info.keys():  
                patch_info[key] = 1
            else:
                patch_info[key] += 1
            return 1
        return total_patches
    
    def _build_tree(self):
        if self.bbox == None:
            h,w = self.domain.shape
            assert h>0 and w >0, "Wrong img size."
            self.bbox = Rect(0,w,0,h)
        value = self.bbox.contains(self.domain)
        if value > self.max_value and self.depth<self.max_depth:
            self.divide()
        
    
    def divide(self):
        x1,x2,y1,y2 = self.bbox.get_coord()
        
        # bbox of the current node.
        self.lt = QuadTree(self.domain, Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2), self.depth + 1, self.max_value, self.max_depth)
        self.rt = QuadTree(self.domain, Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2), self.depth + 1, self.max_value, self.max_depth)
        self.lb = QuadTree(self.domain, Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2)), self.depth + 1, self.max_value, self.max_depth)
        self.rb = QuadTree(self.domain, Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2)), self.depth + 1, self.max_value, self.max_depth)
                                  
        self.divided = True
        self.childs = [self.lt, self.rt, self.lb, self.rb]
    
    def draw(self, ax, c='grey', lw=1, **kwargs):
        self.bbox.draw(ax=ax)
        if self.divided:
            for c_n in self.childs:
                c_n.draw(ax=ax)
                
