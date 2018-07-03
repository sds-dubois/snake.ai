"""
Manage GUI
"""

import pygame, interface
import collections
from pdb import set_trace as t


class Options:
    def __init__(self):
        # Colors
        self.colors = {'BLACK' : (0, 0, 0), 'WHITE' : (255,255,255), 'RED' : (255,0,0), 'GREEN':(0,255,0),\
                       'BLUE':(0,0,255),'BRONZE':(205,127,50), 'GRAY':(180,180,180), 'GOLD':(212,175,55),\
                       'VIOLET' : (200,0,255)}
        self.snake_colors = ['RED','GREEN','BLUE','VIOLET','GRAY']
        self.candy_colors = ['BRONZE','GOLD']
        self.headColor = 'WHITE'
        # Segment geometry        
        self.segment_side = 10
        self.segment_margin = 2
        self.total_size = self.segment_side + self.segment_margin


class SegmentSprite(pygame.sprite.Sprite):
    def __init__(self, position,rgb_color, options):
        # Call the parent's constructor
        super(SegmentSprite,self).__init__() 
        # Set height, width
        self.image = pygame.Surface([options.segment_side, options.segment_side])
        self.image.fill(rgb_color)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.x = position[0]
        self.rect.y = position[1]


class SnakeSprite:
    def __init__(self,positions,color,options):
        head = positions[0]
        tail = positions[1:]   
        # Create list of all segment objects
        segments = [SegmentSprite(head, options.colors[options.headColor], options)] 
        pos_count = collections.defaultdict(int)
        for pos in tail:
            pos_count[pos] +=1
        for pos,n in pos_count.iteritems():
            tile_color = options.colors[color]
            if n>1:
                tile_color = self.darken(tile_color)                    
            segments.append(SegmentSprite(pos, tile_color, options))
        self.segments = segments

    def darken(self,rgb_color):
        mu = 0.6
        return tuple([int(c * mu) for c in rgb_color])

class Window:
    def __init__(self,grid_size,title,options): 
        self.options = options       
        self.size = 2*[grid_size*options.total_size]
        self.title = title
        self.all_sprites = pygame.sprite.Group()
        self.display = pygame.display
        self.screen = self.display.set_mode(self.size)
        self.display.set_caption(title) 
        
    def updateSprites(self,state):
        self.all_sprites = pygame.sprite.Group() 

        # Show candies
        for pos,value in state.candies.items():
            color = 'GOLD' if value == interface.CANDY_BONUS else 'BRONZE'
            self.all_sprites.add(SegmentSprite(self.xy2uv([pos])[0],self.options.colors[color],self.options))

        # Show Snakes
        for i,snake in state.snakes.items():
            color = self.options.snake_colors[i % len(self.options.snake_colors)] 
            self.all_sprites.add(SnakeSprite(self.xy2uv(snake.position),color,self.options).segments)

    def refresh(self):
        # -- Draw everything
        # Clear screen
        self.screen.fill(self.options.colors['BLACK'])
        self.all_sprites.draw(self.screen)    
        # Flip screen
        self.display.flip()

    def print_message(self, message):
        print message

    # Convert grid (x,y) gui/pixel (u,v) indices
    def xy2uv(self,xy_list):        
        uv_list = [None]*len(xy_list)
        for i,xy in enumerate(xy_list):
            x,y = xy
            uv_list[i] = (x * self.options.total_size, y * self.options.total_size)
        return uv_list

