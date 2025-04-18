// Ugly solution for loading js via folium
function(e1){
    if (type === 'rectangle') {
        if (window.last_drawn_bbox) {
            drawnItems.removeLayer(window.last_drawn_bbox);
        }
        // Store the last drawn rectangle
        window.last_drawn_bbox = e1.target;
        // Get the bounds of the shape (rectangle or polygon)
        var bounds = layer.getBounds();
        // Extract southwest and northeast corners
        var sw = bounds.getSouthWest();
        var ne = bounds.getNorthEast();
        // Create HTML content for the tooltip
        var tooltipContent = `
            <table style="border-collapse: collapse; width: 150px;">
                <tr><th colspan="2" style="border-bottom: 1px solid #ddd; padding: 4px;">Area Coordinates</th></tr>
                <tr><td style="padding: 4px;">SW</td><td style="padding: 4px;">${sw.lat.toFixed(5)}, ${sw.lng.toFixed(5)}</td></tr>
                <tr><td style="padding: 4px;">NE</td><td style="padding: 4px;">${ne.lat.toFixed(5)}, ${ne.lng.toFixed(5)}</td></tr>
            </table>
        `;
        e1.target.bindTooltip(tooltipContent);
    } else if (type === 'marker') {
        if (window.last_drawn_hotel_point) {
            drawnItems.removeLayer(window.last_drawn_hotel_point);
        }
        // Store the last drawn hotel point
        window.last_drawn_hotel_point = e1.target;
        // Create HTML content for the tooltip
        var latlng = layer.getLatLng();
        var tooltipContent = `
            <table style="border-collapse: collapse; width: 150px;">
                <tr><th colspan="2" style="border-bottom: 1px solid #ddd; padding: 4px;">Potential Hotel Site Marker</th></tr>
                <tr><td style="padding: 4px;">Lat</td><td style="padding: 4px;">${latlng.lat.toFixed(5)}</td></tr>
                <tr><td style="padding: 4px;">Lon</td><td style="padding: 4px;">${latlng.lng.toFixed(5)}</td></tr>
            </table>
        `;
        e1.target.bindTooltip(tooltipContent);
    }
}