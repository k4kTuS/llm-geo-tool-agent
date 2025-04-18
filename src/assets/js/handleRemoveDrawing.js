function(e1){
    if (type === 'rectangle') {
        if (e1.target === window.last_drawn_bbox) {
            window.last_drawn_bbox = null;
        }
    } else if (type === 'marker') {
        if (e1.target === window.last_drawn_hotel_point) {
            window.last_drawn_hotel_point = null;
        }
    }
}