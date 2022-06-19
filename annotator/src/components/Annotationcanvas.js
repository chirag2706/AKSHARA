import { useEffect, useRef } from "react";
import { addAnnotations, deleteAnnotation } from "../scripts/document";
import "../styles/popup.css";

const Annotationcanvas = ({ data }) => {
    let ref = useRef();
    let { annotations, setAnnotations, documentImage, token, projectId, documentId } = data;

    useEffect(() => {
        let canvas = ref.current;
        let ctx = canvas.getContext("2d");
        let shapes = [...annotations];
        let tempShape = {};
        let isDragging = false;

        let image = new Image();
        image.src = "data:image/png;base64," + documentImage;
        image.onload = () => {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
            shapes.unshift({
                img: image,
                type: "image",
                visible: true,
            });
            drawAll();
        };

        function reOffset() {
            var BB = canvas.getBoundingClientRect();
            offsetX = BB.left;
            offsetY = BB.top;
        }
        var offsetX, offsetY;
        reOffset();

        canvas.onmousedown = handleMouseDown;
        canvas.onmousemove = handleMouseMove;
        canvas.onmouseup = handleMouseUp;
        canvas.onmouseout = handleMouseOut;

        function drawAll() {
            let cw = canvas.width;
            let ch = canvas.height;
            ctx.clearRect(0, 0, cw, ch);
            for (var i = 0; i < shapes.length; i++) {
                var shape = shapes[i];
                if (shape.visible) {
                    if (shape.type === "image") {
                        ctx.drawImage(shape.img, 0, 0);
                    } else {
                        ctx.lineWidth = 2;
                        if(shape.isAntiPattern) {
                            ctx.strokeStyle = "red";
                        } else if(shape.groundTruth) {
                            ctx.strokeStyle = "green";
                        } else {
                            ctx.strokeStyle = "yellow";
                        }
                        ctx.strokeRect(shape.x, shape.y, shape.width, shape.height);
                    }
                }
            }
        }

        function collides(x, y) {
            var isCollision = false;
            for (var i = 0, len = shapes.length; i < len; i++) {
                if (shapes[i].type === "rectangle") {
                    var left = shapes[i].x,
                        right = shapes[i].x + shapes[i].width;
                    var top = shapes[i].y,
                        bottom = shapes[i].y + shapes[i].height;
                    if (right >= x && left <= x && bottom >= y && top <= y) {
                        isCollision = i;
                    }
                }
            }
            return isCollision;
        }

        canvas.addEventListener("click", (e) => {
            var rect = collides(e.offsetX, e.offsetY);
            if (rect) {
                launchPopup(rect, offsetX, offsetY);
            }
        });

        function handleMouseDown(e) {
            e.preventDefault();
            e.stopPropagation();

            reOffset();
            tempShape.x = parseInt(e.clientX - offsetX);
            tempShape.y = parseInt(e.clientY - offsetY);
            tempShape.type = "rectangle";
            tempShape.visible = true;

            isDragging = true;
        }

        function handleMouseUp(e) {
            reOffset();
            if (!isDragging) {
                return;
            }

            e.preventDefault();
            e.stopPropagation();

            if (tempShape.width && tempShape.height) shapes.push(tempShape);
            tempShape = {};
            isDragging = false;
        }

        function handleMouseOut(e) {
            reOffset();

            if (!isDragging) {
                return;
            }

            e.preventDefault();
            e.stopPropagation();

            if (tempShape.width && tempShape.height) shapes.push(tempShape);
            tempShape = {};
            isDragging = false;
        }

        function handleMouseMove(e) {
            reOffset();
            if (!isDragging) {
                return;
            }

            e.preventDefault();
            e.stopPropagation();
            drawAll();

            let mouseX = parseInt(e.clientX - offsetX);
            let mouseY = parseInt(e.clientY - offsetY);

            var dx = mouseX - tempShape.x;
            var dy = mouseY - tempShape.y;

            ctx.strokeRect(tempShape.x, tempShape.y, dx, dy);
            tempShape.width = dx;
            tempShape.height = dy;
        }

        //pop up handlers
        let selectedRectangle;
        let popup = document.getElementById("pop");
        let popname = document.getElementById("pop-name");
        let popcheck = document.getElementById("pop-check");

        popname.onchange = (e) => {
            e.preventDefault();
            shapes[selectedRectangle].name = e.target.value ?? "No name";
        };

        popcheck.onclick = (e) => {
            shapes[selectedRectangle].isAntiPattern = e.target.checked;
        };

        document.getElementById("pop-save").onclick = async (e) => {
            e.preventDefault();
            let lannotations = shapes.filter((shape) => {
                return shape.type === "rectangle" && shape.name ? true : false;
            });
            let shape = shapes[selectedRectangle];
            shape.groundTruth = true
            let rect = {
                project: projectId,
                document: documentId,
                annotation: {
                    _id: shape._id ?? "",
                    name: shape.name,
                    topX: shape.x,
                    topY: shape.y,
                    bottomX: shape.x + shape.width,
                    bottomY: shape.y + shape.height,
                    is_antipattern: shape.isAntiPattern,
                },
            };
            let response = await addAnnotations(token, rect);
            if (response) {
                setAnnotations(lannotations);
            }
            popup.style.display = "none";
        };

        document.getElementById("pop-delete").onclick = async (e) => {
            e.preventDefault();
            if (shapes[selectedRectangle]._id) {
                let response = await deleteAnnotation(token, {
                    project: projectId,
                    document: documentId,
                    annotation: {
                        _id: shapes[selectedRectangle]._id,
                        user: shapes[selectedRectangle].user
                    },
                });
                if (response) {
                    shapes.splice(selectedRectangle, 1);
                    drawAll();
                    let lannotations = shapes.filter((shape) => {
                        return shape.type === "rectangle" ? true : false;
                    });
                    setAnnotations(lannotations);
                    popup.style.display = "none";
                }
            } else {
                shapes.splice(selectedRectangle, 1);
                drawAll();
                let lannotations = shapes.filter((shape) => {
                    return shape.type === "rectangle" ? true : false;
                });
                setAnnotations(lannotations);
                popup.style.display = "none";
            }
        };

        document.getElementById("pop-close").onclick = (e) => {
            e.preventDefault();
            popup.style.top = 0;
            popup.style.left = 0;
            popup.style.display = "none";
        };

        function launchPopup(i, offx, offy) {
            selectedRectangle = i;
            let shape = shapes[i];
            popup.style.display = "block";
            popup.style.top = offy + shape.y + shape.height + "px";
            popup.style.left = offx + shape.x + shape.width + "px";
            popname.value = shape.name ?? "";
            popcheck.checked = shape.isAntiPattern ?? false;
        }
    }, [annotations]);

    return (
        <div>
            <div id="pop" className="canvas-popup">
                <div className="mb-3">
                    <label htmlFor="pop-name" className="form-label">
                        Name
                    </label>
                    <input type="text" className="form-control" id="pop-name" aria-describedby="emailHelp" required></input>
                </div>
                <div className="mb-3 form-check">
                    <input type="checkbox" className="form-check-input" id="pop-check"></input>
                    <label className="form-check-label" htmlFor="pop-check">
                        Mark as anti-pattern
                    </label>
                </div>
                <button type="submit" id="pop-save" className="btn btn-success">
                    Save
                </button>
                <button id="pop-delete" className="btn btn-danger">
                    Delete
                </button>
                <button id="pop-close" className="btn btn-primary">
                    Close
                </button>
            </div>
            <div
                style={{
                    float: "left",
                    width: "100%",
                    height: "800px",
                    overflow: "scroll",
                    border: "1px solid black",
                    margin: "15px",
                }}
            >
                <canvas ref={ref}></canvas>
            </div>
        </div>
    );
};

export default Annotationcanvas;
