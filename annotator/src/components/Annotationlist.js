import { useEffect } from "react";
import { clearPredictedAnnotations } from "../scripts/document";

const Annotationlist = ({ data }) => {
    let { annotations, setAnnotations, projectId, documentId, token } = data;
    let objmap = new Map();
    useEffect(() => {}, [annotations]);

    for (var i = 0; i < annotations.length; i++) {
        let anno = annotations[i];
        let annocategory = objmap.get(anno.name);
        if (
            annocategory === undefined ||
            annocategory === null ||
            annocategory === "undefined"
        ) {
            objmap.set(anno.name, [anno]);
        } else {
            objmap.set(anno.name,[...annocategory, anno]);
        }
    }

    function disableAnnotations(e, index){
        setAnnotations(annotations.map((value)=>{
            if(value.name === index){
                return {
                    ...value,
                    visible: e.target.checked
                }
            }
            return value;
        }))
    }

    async function clearAllPredictedAnnotations(index){
        let status = await clearPredictedAnnotations(token, {
            'project': projectId,
            'document': documentId,
            'name': index
        });
        if(status) {
            setAnnotations(annotations.filter((value)=>{
                if((value.name === index)){
                    if(value.groundTruth === true) {
                        return value
                    }
                } else {
                    return value
                }
            }))
        }
    }

    function getUi() {
        let components = [];
        objmap.forEach((value, index) => {
            let c = (
                <div style={{marginTop:"8px", border:"1px dashed black", padding:"6px"}} key={index}>
                    <input type="checkbox" style={{ display: "inline" }} onChange={(e) => {disableAnnotations(e,index)}} checked={value[0].visible}></input>
                    <div style={{ display: "inline", marginLeft: "15px" }}>
                        <strong>{"Name: " + index}</strong>
                    </div>
                    <div >
                        <i>{"Count: " + value.length.toString()}</i>
                        <div className="moreinfo-tag" onClick={() => {clearAllPredictedAnnotations(index)}}>
                            clear all
                        </div>
                    </div>
                </div>
            );
            components.push(c);
        });
        return components;
    }

    return (
        <div style={{ maxHeight: "400px", overflow: "auto" }}>
            {getUi()}
        </div>
    );
};

export default Annotationlist;
